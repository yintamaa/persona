package assistant

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"sync"
	"testing"
	"time"

	"github.com/nwverse/persona/server/internal/ai"
	"github.com/nwverse/persona/server/internal/i18n"
	"github.com/nwverse/persona/server/internal/proto"
	"github.com/nwverse/persona/server/internal/rpc"
)

// runClock is a clock the tests move by hand. It starts at the real time, so a deadline made from it is in the real
// future and no context expires while a test runs.
type runClock struct {
	mu  sync.Mutex
	now time.Time
}

func newRunClock() *runClock { return &runClock{now: time.Now()} }

func (c *runClock) Now() time.Time {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.now
}

func (c *runClock) advance(d time.Duration) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.now = c.now.Add(d)
}

// slowRunDevs is a node whose model.run takes take on the clock, then answers or reports its run's timeout.
type slowRunDevs struct {
	viaDevs
	clock *runClock

	mu     sync.Mutex
	take   time.Duration
	answer bool
	runs   []proto.ModelRunParams
}

func (d *slowRunDevs) next(take time.Duration, answer bool) {
	d.mu.Lock()
	defer d.mu.Unlock()
	d.take, d.answer = take, answer
}

func (d *slowRunDevs) ran() int {
	d.mu.Lock()
	defer d.mu.Unlock()
	return len(d.runs)
}

func (d *slowRunDevs) CallNode(_ context.Context, id, method string, params json.RawMessage) (json.RawMessage, error) {
	if method != proto.MethodModelRun || id != "server" {
		return nil, fmt.Errorf("unexpected %s on %s", method, id)
	}
	var p proto.ModelRunParams
	if err := json.Unmarshal(params, &p); err != nil {
		return nil, err
	}
	d.mu.Lock()
	d.runs = append(d.runs, p)
	take, answer := d.take, d.answer
	d.mu.Unlock()
	d.clock.advance(take)
	if !answer {
		return nil, &rpc.Error{Code: proto.CodeModelRunTimeout, Message: "model.run: timed out"}
	}
	return json.Marshal(proto.ModelRunAnswer{Text: "fix-login", Model: "m-run"})
}

// launcherFloorService is a manager whose kb uses (tab names, to-do and conversation titles) run on claude on
// "server" with a model of its own (the launcher gates are process-wide), on a slow node and a clock of its own; it
// counts the calls it would show in Settings › Models.
func launcherFloorService(t *testing.T) (*Service, *slowRunDevs, *usageStore, *runClock) {
	t.Helper()
	f := newFakeHub().withMachine()
	f.ownModels = true
	st := &usageStore{downStore: &downStore{memStore: &memStore{}}}
	s := newServiceOn(t, f, st)
	clock := newRunClock()
	s.runFloors.now = clock.Now
	devs := &slowRunDevs{clock: clock}
	s.devs = devs
	model := fmt.Sprintf("m-floor-%s-%d", t.Name(), time.Now().UnixNano())
	if r := modelRequest(s, "PATCH", "kb", fmt.Sprintf(`{"revision":0,"provider":"launcher.claude","model":%q}`, model)); r.Code != 200 {
		t.Fatalf("pick claude: %d %s", r.Code, r.Body.String())
	}
	return s, devs, st, clock
}

func (u *usageStore) counted() int {
	u.mu.Lock()
	defer u.mu.Unlock()
	return len(u.calls)
}

// askWithin is one agent.title request whose caller can wait wait on the service's clock.
func askWithin(s *Service, clock *runClock, wait time.Duration) error {
	ctx, cancel := context.WithDeadline(context.Background(), clock.Now().Add(wait))
	defer cancel()
	_, err := s.purposeModel("agent.title").Chat(ctx, ai.Request{Messages: []ai.Message{{Role: ai.RoleUser, Content: "fix the login"}}})
	return err
}

// A request that cannot wait as long as a run on its launcher takes fails at once, says why, starts nothing and is no
// model call; the first request tries, one that fits always runs and its answer lowers the floor, and an hour after
// the last run the launcher is tried again (#13443).
func TestLauncherRunHeldBackWhenItCannotAnswerInTime(t *testing.T) {
	s, devs, st, clock := launcherFloorService(t)

	// Nothing known: the first request tries, and its unanswered 25 s are what a run takes for now.
	devs.next(25*time.Second, false)
	if err := askWithin(s, clock, 25*time.Second); !errors.Is(err, context.DeadlineExceeded) || runHeldBack(err) {
		t.Fatalf("first request = %v; want a timeout that ran", err)
	}
	if devs.ran() != 1 || st.counted() != 1 {
		t.Fatalf("first request: %d runs, %d counted; want 1, 1", devs.ran(), st.counted())
	}

	// A tab name's 8 s, and 20 s, cannot cover 25 s: refused at once, uncounted, and not a timeout or transient.
	for _, wait := range []time.Duration{8 * time.Second, 20 * time.Second} {
		before := clock.Now()
		err := askWithin(s, clock, wait)
		if !runHeldBack(err) || errors.Is(err, context.DeadlineExceeded) || ai.Transient(err) || ai.IsPaused(err) {
			t.Fatalf("%s request = %v; want held back, no timeout", wait, err)
		}
		want := i18n.T(s.lang(context.Background()), "assistant-models.err_run_too_slow",
			i18n.Args{"name": "server", "launcher": "claude", "seconds": 25, "wait": int(wait.Seconds())})
		if err.Error() != want {
			t.Fatalf("message = %q; want %q", err, want)
		}
		if !clock.Now().Equal(before) {
			t.Fatalf("a held-back request waited %s", clock.Now().Sub(before))
		}
	}
	if devs.ran() != 1 || st.counted() != 1 {
		t.Fatalf("held back: %d runs, %d counted; want still 1, 1", devs.ran(), st.counted())
	}

	// A request that can wait runs; its 12 s answer lowers the floor, so 15 s fits now and 8 s still does not.
	devs.next(12*time.Second, true)
	if err := askWithin(s, clock, 50*time.Second); err != nil {
		t.Fatalf("a request that fits: %v", err)
	}
	if err := askWithin(s, clock, 15*time.Second); err != nil {
		t.Fatalf("15 s after a 12 s answer: %v", err)
	}
	if err := askWithin(s, clock, 8*time.Second); !runHeldBack(err) {
		t.Fatalf("8 s after a 12 s answer = %v; want held back", err)
	}
	if devs.ran() != 3 || st.counted() != 3 {
		t.Fatalf("after the answers: %d runs, %d counted; want 3, 3", devs.ran(), st.counted())
	}

	// Held-back requests do not keep the floor: an hour after the last run, the tab name tries again.
	clock.advance(30 * time.Minute)
	if err := askWithin(s, clock, 8*time.Second); !runHeldBack(err) {
		t.Fatalf("half an hour later = %v; want held back", err)
	}
	clock.advance(30 * time.Minute)
	devs.next(5*time.Second, true)
	if err := askWithin(s, clock, 8*time.Second); err != nil {
		t.Fatalf("an hour after the last run: %v", err)
	}
	if devs.ran() != 4 || st.counted() != 4 {
		t.Fatalf("after the hour: %d runs, %d counted; want 4, 4", devs.ran(), st.counted())
	}
}

// A held-back request never reaches the launcher's gate (ai.NewRun): it is no run that answered, so it does not end
// a series of timeouts that holds a wedged launcher.
func TestHeldBackRunIsUnseenByTheLauncherGate(t *testing.T) {
	ai.SetBusyPauseForTest(t, time.Hour)
	s, devs, _, clock := launcherFloorService(t)
	b, err := s.resolveModel(context.Background(), "agent.title")
	if err != nil {
		t.Fatal(err)
	}
	cfg, err := s.apiConfig(context.Background(), b)
	if err != nil || cfg.RunLauncher != "claude" {
		t.Fatalf("agent.title runs on %+v (%v)", cfg, err)
	}

	devs.next(20*time.Second, false)
	if err := askWithin(s, clock, time.Minute); !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("first timeout = %v", err)
	}
	if err := askWithin(s, clock, 8*time.Second); !runHeldBack(err) {
		t.Fatalf("short request = %v; want held back", err)
	}
	if err := askWithin(s, clock, time.Minute); !ai.IsPaused(err) {
		t.Fatalf("second timeout = %v; want the gate to hold the launcher", err)
	}
	if p := ai.Held(cfg); p == nil || p.Reason != ai.ReasonBusy {
		t.Fatalf("Held = %+v; want busy", p)
	}
}

// The floor book on its own: keys, which observation wins, the slack after an unanswered wait and the hour.
func TestRunFloors(t *testing.T) {
	var b runFloors
	at := time.Unix(1_800_000_000, 0)
	k := runFloorKey{machine: "m", launcher: "agy", model: "flash", effort: "off"}
	cfg := ai.Config{ViaMachine: "m", RunLauncher: "agy", Model: "flash", ReasoningEffort: "off"}
	if runFloorKeyOf(cfg) != k {
		t.Fatalf("key of %+v = %+v", cfg, runFloorKeyOf(cfg))
	}
	held := func(k runFloorKey, wait time.Duration, now time.Time) bool {
		_, h := b.heldBack(k, wait, now)
		return h
	}
	if held(k, time.Millisecond, at) {
		t.Fatal("nothing known held a request back")
	}
	b.ran(k, 8*time.Second, false, at)
	if f, h := b.heldBack(k, 8*time.Second, at); !h || f != 8*time.Second {
		t.Fatalf("8 s after an unanswered 8 s: floor %s, held %v", f, h)
	}
	if !held(k, 8*time.Second+500*time.Millisecond, at) || held(k, 9*time.Second, at) {
		t.Fatal("an unanswered wait holds back what can wait at most a second longer, and no more")
	}
	b.ran(k, 20*time.Second, false, at)
	b.ran(k, 10*time.Second, false, at)
	if f, _ := b.heldBack(k, 0, at); f != 20*time.Second {
		t.Fatalf("floor after waits of 8, 20 and 10 s = %s; want the longest", f)
	}
	b.ran(k, 12*time.Second, true, at)
	b.ran(k, 30*time.Second, true, at)
	b.ran(k, 40*time.Second, false, at)
	if f, h := b.heldBack(k, 12*time.Second, at); h || f != 12*time.Second {
		t.Fatalf("after answers in 12 and 30 s: floor %s, held %v; want the fastest, 12 s fits", f, h)
	}
	if !held(k, 11*time.Second, at) {
		t.Fatal("11 s fits a 12 s answer")
	}
	for _, other := range []runFloorKey{
		{machine: "n", launcher: "agy", model: "flash", effort: "off"},
		{machine: "m", launcher: "claude", model: "flash", effort: "off"},
		{machine: "m", launcher: "agy", model: "pro", effort: "off"},
		{machine: "m", launcher: "agy", model: "flash", effort: "high"},
	} {
		if held(other, time.Millisecond, at) {
			t.Fatalf("%+v held back by another key's runs", other)
		}
	}
	if !held(k, time.Second, at.Add(runFloorTTL-time.Second)) || held(k, time.Second, at.Add(runFloorTTL)) {
		t.Fatal("a floor lasts an hour after the last run, and no longer")
	}
	if _, ok := b.floors[k]; ok {
		t.Fatal("an expired floor stayed")
	}
	b.ran(k, 8*time.Second, false, at)
	b.ran(runFloorKey{machine: "x"}, time.Second, true, at.Add(runFloorTTL))
	if _, ok := b.floors[k]; ok || len(b.floors) != 1 {
		t.Fatalf("a run did not drop the floors past their hour: %v", b.floors)
	}
}

// The book is shared by every purpose's requests at once; run with -race.
func TestRunFloorsAcrossGoroutines(t *testing.T) {
	var b runFloors
	at := time.Unix(1_800_000_000, 0)
	k := runFloorKey{machine: "m", launcher: "agy"}
	var wg sync.WaitGroup
	for i := range 16 {
		wg.Go(func() {
			for j := range 100 {
				b.ran(k, time.Duration(i*100+j+1)*time.Millisecond, j%2 == 0, at)
				b.heldBack(k, time.Second, at)
			}
		})
	}
	wg.Wait()
	if f, _ := b.heldBack(k, 0, at); f != time.Millisecond {
		t.Fatalf("fastest answer = %s; want 1ms", f)
	}
}

// A to-do title made in the background waits 50 s on a launcher (a cold CLI), inside nameTodoLater's minute; on an API
// connection, and where a caller waits, it keeps 15 s.
func TestLaterTodoTitleWaitsForACLIOnALauncher(t *testing.T) {
	api := newServiceOn(t, func() *fakeHub {
		f := newFakeHub().withMachine()
		f.ownModels = true
		f.hello.Models.AI = ai.Config{BaseURL: "http://127.0.0.1:1", APIKey: "test-key", Model: "relay-model"}
		return f
	}(), &downStore{memStore: &memStore{}})
	if got := api.laterTodoTitleTimeout(context.Background()); got != 15*time.Second {
		t.Fatalf("on an API connection: %s; want 15s", got)
	}

	s, devs, _, _ := launcherFloorService(t)
	if got := s.laterTodoTitleTimeout(context.Background()); got != 50*time.Second {
		t.Fatalf("on a launcher: %s; want 50s", got)
	}
	devs.next(0, true)
	s.nameTodoLater(context.Background(), "no-such-todo", "from", "fix the login crash on the phone")
	s.todos.titling.Wait()
	if s.todoTitleFromModel(context.Background(), "fix the login crash") == "" {
		t.Fatal("no title where a caller waits")
	}
	devs.mu.Lock()
	defer devs.mu.Unlock()
	if len(devs.runs) != 2 {
		t.Fatalf("%d runs; want 2", len(devs.runs))
	}
	if ms := devs.runs[0].TimeoutMS; ms <= 15_000 || ms > 50_000 {
		t.Fatalf("background to-do title on a launcher waits %d ms; want 50 s", ms)
	}
	if ms := devs.runs[1].TimeoutMS; ms > 15_000 {
		t.Fatalf("a to-do title its caller waits for waits %d ms; want 15 s", ms)
	}
}

// A conversation title its launcher is too slow for keeps a stand-in that a later turn names again; it is not final.
func TestHeldBackConversationTitleIsAStandIn(t *testing.T) {
	s, devs, _, _ := launcherFloorService(t)
	s.runFloors.now = nil // makeTitle's own 45 s deadline is on the real clock
	b, _ := s.resolveModel(context.Background(), "assistant.title")
	cfg, err := s.apiConfig(context.Background(), b)
	if err != nil || cfg.RunLauncher == "" {
		t.Fatalf("assistant.title runs on %+v (%v)", cfg, err)
	}
	s.runFloors.ran(runFloorKeyOf(cfg), 2*time.Minute, false, time.Now())
	if _, final := s.makeTitle(context.Background(), time.Now(), "fix the login crash", "On it.", ""); final {
		t.Fatal("a held-back title was taken as final")
	}
	if devs.ran() != 0 {
		t.Fatalf("%d runs for a held-back title", devs.ran())
	}
}

// A background Agent name its launcher is too slow for counts as no failure: no failure cooldown, no failure limit.
func TestHeldBackAgentNameIsNoFailure(t *testing.T) {
	s, f, calls := namingFixture(t, func(w http.ResponseWriter, r *http.Request) { namingReply(w, "unused") })
	if r := modelRequest(s, "PATCH", "inbox.triage", `{"revision":0,"provider":"launcher.claude"}`); r.Code != 200 {
		t.Fatalf("pick claude: %d %s", r.Code, r.Body.String())
	}
	// No node: a run that started would fail, and count as a failure, without one.
	b, _ := s.resolveModel(context.Background(), "inbox.triage")
	cfg, err := s.apiConfig(context.Background(), b)
	if err != nil || cfg.RunLauncher != "claude" {
		t.Fatalf("inbox.triage runs on %+v (%v)", cfg, err)
	}
	s.runFloors.ran(runFloorKeyOf(cfg), 2*time.Minute, false, time.Now())

	c := f.sessions["server"][0]
	if !s.tryAgentName(context.Background(), "server", c, time.Now()) {
		t.Fatal("no naming attempt")
	}
	s.agentNames.mu.Lock()
	r := s.agentNames.records[agentNameKey("server", c.SessionID)]
	s.agentNames.mu.Unlock()
	if r.Failures != 0 || r.LastFailure != "" {
		t.Fatalf("a held-back name counted as a failure: %+v", r)
	}
	if calls.Load() != 0 {
		t.Fatalf("held back, yet %d API calls", calls.Load())
	}
}
