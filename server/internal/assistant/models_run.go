package assistant

import (
	"cmp"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"slices"
	"sync"
	"time"

	"github.com/nwverse/persona/server/internal/ai"
	"github.com/nwverse/persona/server/internal/apierr"
	"github.com/nwverse/persona/server/internal/devices"
	"github.com/nwverse/persona/server/internal/i18n"
	"github.com/nwverse/persona/server/internal/prompts"
	"github.com/nwverse/persona/server/internal/proto"
	"github.com/nwverse/persona/server/internal/rpc"
)

// An API purpose may pick a launcher without a gateway the manager can call (plain claude, codex, agy, or a computer's
// own launcher): each request then runs that launcher's CLI headless once on the main chat's computer, which answers
// with the final text (docs/assistant/model-config.md "Running API purposes on a launcher", #5823, #12027). The manager
// still resolves the purpose and reads the answer; the node runs the CLI with no tools and leaves no session (model.run).

// modelRunTimeout bounds a run whose caller set no deadline.
const modelRunTimeout = 2 * time.Minute

// runClient is the client of c, a launcher on c.ViaMachine (c.RunLauncher). A request its caller cannot wait for
// (runFloors) fails before the launcher's gate (ai.NewRun) admits it: nothing starts, so the gate learns nothing from it.
func (s *Service) runClient(c ai.Config) *ai.Client {
	run := ai.NewRun(c, func(ctx context.Context, r ai.Request) (ai.Reply, error) { return s.modelRun(ctx, c, r) })
	return ai.NewResolved(func(ctx context.Context) (*ai.Client, error) {
		if err := s.runFits(ctx, c); err != nil {
			return nil, err
		}
		return run, nil
	})
}

// Every request on a launcher cold-starts its CLI, and agy's start alone takes longer than a new Agent's tab name may
// wait (8 s): each such run was cancelled unanswered, its caller waited out the whole budget, and the run was wasted
// (#13443). So the manager remembers, per computer, launcher, model and effort, how fast a run there can be: the
// fastest answer, else the longest wait that ended unanswered. A request whose caller cannot wait that long fails at
// once, saying why (runTooSlowError): no CLI starts, and it is no model call (countedClient). A request that fits
// always runs, so its answer can lower the floor; with nothing known the first one tries; and what a key showed is
// forgotten an hour after its last run, so a request held back never keeps a launcher out longer than that.
const (
	runFloorTTL = time.Hour
	// runMissSlack: a wait that ended unanswered says the run needs longer than it. The next request of the same
	// caller has the same budget give or take the milliseconds spent before the run, so one that can wait at most
	// this much longer is held back too.
	runMissSlack = time.Second
)

// runFloorKey is what a run's speed depends on: the computer, the launcher, the model and the effort.
type runFloorKey struct{ machine, launcher, model, effort string }

func runFloorKeyOf(c ai.Config) runFloorKey {
	return runFloorKey{machine: c.ViaMachine, launcher: c.RunLauncher, model: c.Model, effort: c.ReasoningEffort}
}

// runFloor is what the runs on one key showed: the fastest answer (when answered) and the longest wait that ended
// unanswered (missed, 0 for none), as of at, when the last of them ended.
type runFloor struct {
	answered bool
	fastest  time.Duration
	missed   time.Duration
	at       time.Time
}

// runFloors holds the runFloor of each key. Every purpose's requests share it from their own goroutines, so mu guards
// the map. now is the clock (nil: time.Now; tests set it before the first request).
type runFloors struct {
	mu     sync.Mutex
	floors map[runFloorKey]runFloor
	now    func() time.Time
}

func (b *runFloors) clock() time.Time {
	if b.now != nil {
		return b.now()
	}
	return time.Now()
}

// heldBack: a request on k that can wait wait at now cannot be answered in time; floor is the time a run there has
// taken. Nothing known, or known more than runFloorTTL ago, holds nothing back.
func (b *runFloors) heldBack(k runFloorKey, wait time.Duration, now time.Time) (floor time.Duration, held bool) {
	b.mu.Lock()
	defer b.mu.Unlock()
	f, ok := b.floors[k]
	switch {
	case !ok:
		return 0, false
	case now.Sub(f.at) >= runFloorTTL:
		delete(b.floors, k)
		return 0, false
	case f.answered:
		return f.fastest, wait < f.fastest
	default:
		return f.missed, wait < f.missed+runMissSlack
	}
}

// ran records a run on k that ended at now after took: answered, or a wait that ended unanswered. Keys whose last run
// is more than runFloorTTL old go.
func (b *runFloors) ran(k runFloorKey, took time.Duration, answered bool, now time.Time) {
	b.mu.Lock()
	defer b.mu.Unlock()
	if b.floors == nil {
		b.floors = map[runFloorKey]runFloor{}
	}
	for key, f := range b.floors {
		if now.Sub(f.at) >= runFloorTTL {
			delete(b.floors, key)
		}
	}
	f := b.floors[k]
	if answered {
		if !f.answered || took < f.fastest {
			f.fastest = took
		}
		f.answered = true
	} else {
		f.missed = max(f.missed, took)
	}
	f.at = now
	b.floors[k] = f
}

// runTooSlowError is a request on a launcher that was not started: its caller could not wait as long as a run there
// takes (runFloors). It is no timeout (nothing ran out of time) and not transient (the same request is held back
// again); runHeldBack tells it apart.
type runTooSlowError struct{ msg string }

func (e *runTooSlowError) Error() string { return e.msg }

// runHeldBack: err is a launcher request that was not started because its caller could not wait long enough.
func runHeldBack(err error) bool {
	var e *runTooSlowError
	return errors.As(err, &e)
}

// runFits is nil when a request on c with ctx's deadline may run, else the runTooSlowError that says why not.
func (s *Service) runFits(ctx context.Context, c ai.Config) error {
	now := s.runFloors.clock()
	wait := modelRunTimeout
	if d, ok := ctx.Deadline(); ok {
		wait = d.Sub(now)
	}
	floor, held := s.runFloors.heldBack(runFloorKeyOf(c), wait, now)
	if !held {
		return nil
	}
	return &runTooSlowError{msg: i18n.T(s.lang(ctx), "assistant-models.err_run_too_slow", i18n.Args{
		"name": s.runMachineName(ctx, c.ViaMachine), "launcher": c.RunLauncher,
		"seconds": int(math.Ceil(floor.Seconds())), "wait": int(math.Round(max(wait, 0).Seconds()))})}
}

// modelRun answers r with one model.run on c's computer.
func (s *Service) modelRun(ctx context.Context, c ai.Config, r ai.Request) (ai.Reply, error) {
	if proto.IsExternalLauncher(c.RunLauncher) {
		return ai.Reply{}, i18n.Err("external-adapter.purpose_unsupported")
	}
	if _, ok := ctx.Deadline(); !ok {
		var cancel context.CancelFunc
		ctx, cancel = context.WithTimeout(ctx, modelRunTimeout)
		defer cancel()
	}
	p := proto.ModelRunParams{Launcher: c.RunLauncher, Model: c.Model, System: r.System, Prompt: runPrompt(r.Messages)}
	effortOK := slices.Contains(proto.EffortLevels, c.ReasoningEffort) ||
		(slices.Contains(proto.CodexEffortLevels, c.ReasoningEffort) && s.runLauncherAgent(ctx, c.ViaMachine, c.RunLauncher) == "codex")
	if effortOK { // "off" / "none" (and Codex-only levels on non-Codex launchers) defer to the launcher's own default
		p.Effort = c.ReasoningEffort
	}
	_, p.Configured = s.configuredLauncher(ctx, c.ViaMachine, c.RunLauncher)
	d, _ := ctx.Deadline()
	p.TimeoutMS = max(time.Until(d).Milliseconds(), 1)
	raw, err := json.Marshal(p)
	if err != nil {
		return ai.Reply{}, err
	}
	if s.devs == nil {
		return ai.Reply{}, s.modelRunErr(ctx, c, devices.ErrNoMachine)
	}
	started := s.runFloors.clock()
	out, err := s.devs.CallNode(ctx, c.ViaMachine, proto.MethodModelRun, raw)
	ended := s.runFloors.clock()
	if err != nil {
		if runTimedOut(err) {
			s.runFloors.ran(runFloorKeyOf(c), ended.Sub(started), false, ended)
		}
		return ai.Reply{}, s.modelRunErr(ctx, c, err)
	}
	var a proto.ModelRunAnswer
	if err := json.Unmarshal(out, &a); err != nil {
		return ai.Reply{}, fmt.Errorf("model.run: bad answer: %w", err)
	}
	s.runFloors.ran(runFloorKeyOf(c), ended.Sub(started), true, ended)
	return ai.Reply{Text: a.Text, Model: a.Model, Usage: a.Usage}, nil
}

// runTimedOut: a model.run ended for want of time, the node's run or the manager's wait (modelRunErr's timeout).
func runTimedOut(err error) bool {
	var re *rpc.Error
	return errors.As(err, &re) && re.Code == proto.CodeModelRunTimeout || errors.Is(err, context.DeadlineExceeded)
}

// runLauncherAgent resolves the CLI agent behind launcher name on machine for model.run effort filtering.
func (s *Service) runLauncherAgent(ctx context.Context, machine, name string) string {
	if l, ok := s.configuredLauncher(ctx, machine, name); ok {
		if agent := cmp.Or(l.AgentType, l.Agent); agent != "" {
			return agent
		}
	}
	for _, l := range s.storedLaunchers(ctx) {
		if l.Name == name {
			return l.AgentType()
		}
	}
	if machine != "" {
		for _, m := range s.zshrcLaunchers(ctx, false) {
			if m.MachineID != machine {
				continue
			}
			for _, l := range m.Launchers {
				if l.Name == name {
					return viewAgent(LauncherView{Name: l.Name, Agent: l.Agent})
				}
			}
		}
	}
	for _, v := range s.launcherViews(ctx) {
		if v.Name == name {
			return viewAgent(v)
		}
	}
	return viewAgent(LauncherView{Name: name})
}

// runPrompt is the conversation as the CLI reads it: a single user turn as it is, several as one transcript.
func runPrompt(turns []ai.Message) string {
	if len(turns) == 1 && turns[0].Role == ai.RoleUser {
		return turns[0].Content
	}
	in := make([]prompts.LauncherRunTurn, len(turns))
	for i, t := range turns {
		in[i] = prompts.LauncherRunTurn{Assistant: t.Role == ai.RoleAssistant, Text: t.Content}
	}
	return prompts.RenderLauncherRunInput(in)
}

// modelRunErr says what went wrong in the user's language: the computer offline or gone, the CLI not logged in there,
// no answer in time. The node's other errors (an old node, what the CLI printed) go as they are.
func (s *Service) modelRunErr(ctx context.Context, c ai.Config, err error) error {
	if errors.Is(err, devices.ErrNoMachine) {
		return errViaMachine(c.ViaMachine)
	}
	args := i18n.Args{"name": s.runMachineName(ctx, c.ViaMachine), "launcher": c.RunLauncher}
	var re *rpc.Error
	switch {
	case errors.Is(err, apierr.ErrOffline), errors.Is(err, rpc.ErrClosed):
		return errors.New(i18n.T(s.lang(ctx), "assistant-models.err_run_offline", args))
	case errors.As(err, &re) && re.Code == proto.CodeModelRunLoggedOut:
		return errors.New(i18n.T(s.lang(ctx), "assistant-models.err_run_logged_out", args))
	case runTimedOut(err):
		return fmt.Errorf("%s: %w", i18n.T(s.lang(ctx), "assistant-models.err_run_timeout", args), context.DeadlineExceeded)
	}
	return err
}

// runMachineName is machine's display name for a model.run message, its id when the hub has none.
func (s *Service) runMachineName(ctx context.Context, machine string) string {
	if ctx.Err() == nil { // the display name is cosmetic; do not make another RPC with an expired context
		if d, ok := s.device(ctx, machine); ok && d.Name != "" {
			return d.Name
		}
	}
	return machine
}
