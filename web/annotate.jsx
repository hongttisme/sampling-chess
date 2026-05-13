// annotate.jsx — PGN Annotator

const { useState, useEffect, useMemo, useRef } = React;

// ── Mock game data ────────────────────────────────────────────────────────
// Morphy vs Duke of Brunswick & Count Isouard, Paris 1858 ("Opera Game").
// 17 moves. Black plays the catastrophic 9...b5?? — perfect for showing the
// annotator's blunder-detection.
const SAMPLE_PGN = `[Event "Casual rapid game"]
[Site "Paris Opera"]
[Date "1858.10.21"]
[White "morphy_legend"]
[Black "you"]
[Result "1-0"]
[TimeControl "30+0"]

1. e4 e5 2. Nf3 d6 3. d4 Bg4 4. dxe5 Bxf3 5. Qxf3 dxe5 6. Bc4 Nf6
7. Qb3 Qe7 8. Nc3 c6 9. Bg5 b5 10. Nxb5 cxb5 11. Bxb5+ Nbd7 12. O-O-O Rd8
13. Rxd7 Rxd7 14. Rd1 Qe6 15. Bxd7+ Nxd7 16. Qb8+ Nxb8 17. Rd8# 1-0`;

// Move-level annotations. Index = ply (0..32). White=even, black=odd.
// v is from WHITE's perspective in [-1, 1].
const ANNOTATIONS = [
  // 1. e4 e5
  { v: 0.06, best: 'e4',   best_v: 0.06, top: [['e4',0.06,0.28],['d4',0.05,0.24],['Nf3',0.05,0.18]], verdict: 'best' },
  { v: 0.07, best: 'e5',   best_v: 0.07, top: [['e5',0.07,0.32],['c5',0.05,0.20],['e6',0.04,0.12]], verdict: 'best' },
  // 2. Nf3 d6 — Philidor, slight passivity
  { v: 0.10, best: 'Nf3',  best_v: 0.10, top: [['Nf3',0.10,0.45],['Bc4',0.06,0.18],['Nc3',0.05,0.12]], verdict: 'best' },
  { v: 0.18, best: 'Nc6',  best_v: 0.08, top: [['Nc6',0.08,0.55],['Nf6',0.06,0.18],['d6',0.18,0.10]], verdict: 'inaccuracy' },
  // 3. d4 Bg4 — committing bishop early
  { v: 0.22, best: 'd4',   best_v: 0.22, top: [['d4',0.22,0.40],['Bc4',0.14,0.22],['Nc3',0.10,0.12]], verdict: 'best' },
  { v: 0.30, best: 'Nf6',  best_v: 0.15, top: [['Nf6',0.15,0.28],['Nd7',0.18,0.18],['Bg4',0.30,0.14]], verdict: 'inaccuracy' },
  // 4. dxe5 Bxf3 — forced trade
  { v: 0.34, best: 'dxe5', best_v: 0.34, top: [['dxe5',0.34,0.55],['Nc3',0.18,0.18],['Bb5',0.12,0.10]], verdict: 'best' },
  { v: 0.32, best: 'Bxf3', best_v: 0.32, top: [['Bxf3',0.32,0.78],['Nxe5',0.65,0.06],['Bh5',0.55,0.04]], verdict: 'best' },
  // 5. Qxf3 dxe5
  { v: 0.30, best: 'Qxf3', best_v: 0.30, top: [['Qxf3',0.30,0.85],['gxf3',0.55,0.08],['Nxf3',0.40,0.04]], verdict: 'best' },
  { v: 0.32, best: 'dxe5', best_v: 0.32, top: [['dxe5',0.32,0.75],['Qxe5',0.55,0.12],['Nd7',0.45,0.06]], verdict: 'best' },
  // 6. Bc4 Nf6
  { v: 0.36, best: 'Bc4',  best_v: 0.36, top: [['Bc4',0.36,0.40],['Nc3',0.28,0.20],['Bb5+',0.22,0.12]], verdict: 'best' },
  { v: 0.40, best: 'Nf6',  best_v: 0.40, top: [['Nf6',0.40,0.45],['Nc6',0.32,0.20],['Qe7',0.55,0.10]], verdict: 'best' },
  // 7. Qb3 Qe7 — Qd7 better, Qe7 passive
  { v: 0.45, best: 'Qb3',  best_v: 0.45, top: [['Qb3',0.45,0.32],['O-O',0.38,0.22],['Nc3',0.30,0.14]], verdict: 'best' },
  { v: 0.65, best: 'Qd7',  best_v: 0.42, top: [['Qd7',0.42,0.30],['Qd6',0.45,0.22],['Qe7',0.65,0.18]], verdict: 'mistake' },
  // 8. Nc3 c6 — defending b7
  { v: 0.68, best: 'Nc3',  best_v: 0.68, top: [['Nc3',0.68,0.38],['Bxf7+',0.50,0.18],['Qxb7',0.55,0.16]], verdict: 'best' },
  { v: 0.70, best: 'c6',   best_v: 0.70, top: [['c6',0.70,0.55],['Qc7',0.78,0.18],['Nbd7',0.80,0.10]], verdict: 'best' },
  // 9. Bg5 b5?? — the famous blunder
  { v: 0.75, best: 'Bg5',  best_v: 0.75, top: [['Bg5',0.75,0.42],['O-O-O',0.65,0.22],['Bxf7+',0.55,0.12]], verdict: 'best' },
  { v: 0.96, best: 'Qc7',  best_v: 0.70, top: [['Qc7',0.70,0.30],['Qd6',0.75,0.22],['b5',0.96,0.10]], verdict: 'blunder' },
  // 10. Nxb5 cxb5 — sacrifice & recapture
  { v: 0.95, best: 'Nxb5', best_v: 0.95, top: [['Nxb5',0.95,0.65],['Bxf6',0.40,0.10],['O-O-O',0.55,0.08]], verdict: 'best' },
  { v: 0.93, best: 'cxb5', best_v: 0.93, top: [['cxb5',0.93,0.85],['Nxe4',-0.20,0.04],['Qd7',-0.40,0.02]], verdict: 'best' },
  // 11. Bxb5+ Nbd7 — only defense
  { v: 0.96, best: 'Bxb5+', best_v: 0.96, top: [['Bxb5+',0.96,0.70],['Qxb5+',0.92,0.18],['O-O-O',0.85,0.06]], verdict: 'best' },
  { v: 0.96, best: 'Nbd7', best_v: 0.96, top: [['Nbd7',0.96,0.95],['Qd7',-0.50,0.02],['Kd7',-0.60,0.01]], verdict: 'best' },
  // 12. O-O-O Rd8 — only move (Rxd1 hopeless)
  { v: 0.98, best: 'O-O-O', best_v: 0.98, top: [['O-O-O',0.98,0.85],['Bxd7+',0.92,0.08],['Bxf6',0.82,0.04]], verdict: 'best' },
  { v: 0.97, best: 'Rd8',  best_v: 0.97, top: [['Rd8',0.97,0.92],['Kd7',-0.40,0.02],['Qd6',0.90,0.03]], verdict: 'best' },
  // 13. Rxd7 Rxd7
  { v: 0.99, best: 'Rxd7', best_v: 0.99, top: [['Rxd7',0.99,0.92],['Bxd7+',0.95,0.05],['Bxf6',0.80,0.02]], verdict: 'best' },
  { v: 0.98, best: 'Rxd7', best_v: 0.98, top: [['Rxd7',0.98,0.95],['Kxd7',-0.55,0.02],['Nxd7',0.96,0.02]], verdict: 'best' },
  // 14. Rd1 Qe6 — only defending move
  { v: 0.99, best: 'Rd1',  best_v: 0.99, top: [['Rd1',0.99,0.70],['Bxd7+',0.95,0.20],['Bxf6',0.85,0.05]], verdict: 'best' },
  { v: 0.99, best: 'Qe6',  best_v: 0.99, top: [['Qe6',0.99,0.85],['Kf8',-0.40,0.04],['Rxd1+',0.97,0.06]], verdict: 'best' },
  // 15. Bxd7+ Nxd7
  { v: 1.00, best: 'Bxd7+', best_v: 1.00, top: [['Bxd7+',1.00,0.90],['Rxd7',0.92,0.06],['Bxf6',0.85,0.02]], verdict: 'best' },
  { v: 1.00, best: 'Nxd7', best_v: 1.00, top: [['Nxd7',1.00,0.85],['Kxd7',-0.95,0.05],['Rxd7',0.98,0.05]], verdict: 'best' },
  // 16. Qb8+ Nxb8
  { v: 1.00, best: 'Qb8+', best_v: 1.00, top: [['Qb8+',1.00,0.95],['Rxd7',0.90,0.03],['Qxe6+',0.95,0.02]], verdict: 'best' },
  { v: 1.00, best: 'Nxb8', best_v: 1.00, top: [['Nxb8',1.00,1.00]], verdict: 'best' },
  // 17. Rd8# — mate
  { v: 1.00, best: 'Rd8#', best_v: 1.00, top: [['Rd8#',1.00,1.00]], verdict: 'best' },
];

// ── PGN parser (extract moves only — chess.js then replays) ────────────────
function parsePgn(pgn) {
  // strip headers and comments
  const body = pgn
    .replace(/\[[^\]]*\]/g, '')   // header tags
    .replace(/\{[^}]*\}/g, '')    // comments
    .replace(/;[^\n]*/g, '')      // semicolon comments
    .replace(/\$\d+/g, '')        // NAGs
    .replace(/\d+\.{1,3}/g, '')   // move numbers
    .replace(/(1-0|0-1|1\/2-1\/2|\*)\s*$/, '')
    .trim();
  return body.split(/\s+/).filter(Boolean);
}

// ── Build position FENs by replaying through chess.js ─────────────────────
function buildPositions(pgn) {
  const sans = parsePgn(pgn);
  const c = new Chess();
  const positions = [{ fen: c.fen(), move: null, san: null }];
  for (let i = 0; i < sans.length; i++) {
    const m = c.move(sans[i]);
    if (!m) break;
    positions.push({ fen: c.fen(), move: { from: m.from, to: m.to }, san: m.san, color: m.color });
  }
  return positions;
}

// ── Header parser ─────────────────────────────────────────────────────────
function parseHeaders(pgn) {
  const out = {};
  for (const m of pgn.matchAll(/\[(\w+)\s+"([^"]*)"\]/g)) {
    out[m[1]] = m[2];
  }
  return out;
}

// ── Verdict styling ───────────────────────────────────────────────────────
const VERDICT_INFO = {
  best:        { label: 'best',         tone: 'best',  glyph: '✓', sentence: 'You played the bot\'s top choice.' },
  ok:          { label: 'good',         tone: 'ok',    glyph: '·', sentence: 'Sound move.' },
  inaccuracy:  { label: 'inaccuracy',   tone: 'inacc', glyph: '?!', sentence: 'Slightly imprecise — there was a stronger move.' },
  mistake:     { label: 'mistake',      tone: 'mist',  glyph: '?', sentence: 'This drops half a pawn or more.' },
  blunder:     { label: 'blunder',      tone: 'blund', glyph: '??', sentence: 'A big tactical or positional loss.' },
};

// ── V trajectory chart ────────────────────────────────────────────────────
function VTrajectory({ annotations, positions, currentPly, onJump, side }) {
  const W = 720, H = 80, PAD = 8;
  const n = annotations.length;
  // Values from chosen side's perspective.
  const vals = annotations.map((a, i) => {
    const v = a.v;
    return side === 'b' ? -v : v;
  });
  const maxAbs = Math.max(0.4, ...vals.map(Math.abs)) * 1.1;
  const x = (i) => PAD + (i / Math.max(1, n - 1)) * (W - PAD * 2);
  const y = (v) => {
    const norm = Math.max(-1, Math.min(1, v / maxAbs));
    return PAD + (H - PAD * 2) * (1 - (norm + 1) / 2);
  };
  const path = vals.map((v, i) => `${i === 0 ? 'M' : 'L'} ${x(i).toFixed(1)} ${y(v).toFixed(1)}`).join(' ');
  // Blunders / mistakes for markers
  const events = annotations.map((a, i) => ({ i, v: vals[i], verdict: a.verdict, color: i % 2 === 0 ? 'w' : 'b' }))
    .filter(e => (e.verdict === 'blunder' || e.verdict === 'mistake') && e.color === side);

  // biggest blunder ply
  const biggest = events.length
    ? events.reduce((a, b) => Math.abs(b.v - 0) > Math.abs(a.v - 0) ? b : a, events[0])
    : null;

  return (
    <div className="v-trajectory">
      <div className="v-traj-head">
        <span className="v-traj-title">V trajectory · annotated as {side === 'w' ? 'white' : 'black'}</span>
        {biggest && (
          <button className="v-traj-jump" onClick={() => onJump(biggest.i + 1)}>
            → biggest blunder (move {Math.floor(biggest.i / 2) + 1})
          </button>
        )}
      </div>
      <svg viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="none">
        <defs>
          <linearGradient id="atgrad" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor="var(--pos)" stopOpacity="0.22" />
            <stop offset="50%" stopColor="var(--pos)" stopOpacity="0.05" />
            <stop offset="50%" stopColor="var(--neg)" stopOpacity="0.05" />
            <stop offset="100%" stopColor="var(--neg)" stopOpacity="0.20" />
          </linearGradient>
        </defs>
        <line x1={PAD} y1={y(0)} x2={W - PAD} y2={y(0)} stroke="var(--border-strong)" strokeDasharray="3 3" />
        <path d={path + ` L ${x(n-1)} ${y(0)} L ${x(0)} ${y(0)} Z`} fill="url(#atgrad)" />
        <path d={path} stroke="var(--ink)" strokeWidth="1.6" fill="none" strokeLinejoin="round" />
        {events.map((e, idx) => (
          <line key={idx}
            x1={x(e.i)} x2={x(e.i)}
            y1={PAD} y2={H - PAD}
            stroke={e.verdict === 'blunder' ? 'var(--neg)' : 'var(--warn)'}
            strokeWidth={e.verdict === 'blunder' ? 2 : 1.2}
            opacity="0.55"
          />
        ))}
        {currentPly >= 1 && currentPly <= n && (
          <line x1={x(currentPly - 1)} x2={x(currentPly - 1)} y1={PAD} y2={H - PAD}
                stroke="var(--accent)" strokeWidth="2" />
        )}
        {/* dot at current */}
        {currentPly >= 1 && currentPly <= n && (
          <circle cx={x(currentPly - 1)} cy={y(vals[currentPly - 1])} r="3.5" fill="var(--accent)" />
        )}
      </svg>
    </div>
  );
}

// ── Top app ───────────────────────────────────────────────────────────────
function App() {
  // intake state
  const [intakeTab, setIntakeTab] = useState('paste');
  const [pgnText, setPgnText] = useState(SAMPLE_PGN);
  const [side, setSide] = useState('b'); // 'w' | 'b' | 'both'
  const [analyzed, setAnalyzed] = useState(false);

  // analysis state
  const [positions, setPositions] = useState([]);
  const [headers, setHeaders] = useState({});
  const [annotations, setAnnotations] = useState([]);
  const [currentPly, setCurrentPly] = useState(0); // 0 = starting pos; 1 = after first move

  // perform analysis (mock — instant since we have annotations baked in)
  const runAnalysis = () => {
    const positions = buildPositions(pgnText);
    setPositions(positions);
    setHeaders(parseHeaders(pgnText));
    setAnnotations(ANNOTATIONS.slice(0, positions.length - 1));
    setCurrentPly(0);
    setAnalyzed(true);
  };

  // keyboard nav
  useEffect(() => {
    if (!analyzed) return;
    const onKey = (e) => {
      if (e.target.tagName === 'TEXTAREA' || e.target.tagName === 'INPUT') return;
      if (e.key === 'ArrowRight') setCurrentPly(p => Math.min(positions.length - 1, p + 1));
      else if (e.key === 'ArrowLeft') setCurrentPly(p => Math.max(0, p - 1));
      else if (e.key === 'Home') setCurrentPly(0);
      else if (e.key === 'End') setCurrentPly(positions.length - 1);
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [analyzed, positions.length]);

  // aggregate stats — only for the chosen side
  const stats = useMemo(() => {
    if (!analyzed) return null;
    const sideAnnots = annotations.filter((_, i) => side === 'both' ? true : (i % 2 === 0 ? side === 'w' : side === 'b'));
    const counts = { best: 0, ok: 0, inaccuracy: 0, mistake: 0, blunder: 0 };
    let totalLoss = 0;
    let biggest = null;
    annotations.forEach((a, i) => {
      const myMove = side === 'both' ? true : (i % 2 === 0 ? side === 'w' : side === 'b');
      if (!myMove) return;
      counts[a.verdict] = (counts[a.verdict] || 0) + 1;
      // value loss for THIS side
      const sign = (i % 2 === 0) ? 1 : -1;
      const loss = Math.max(0, (a.best_v - a.v) * sign);
      totalLoss += loss;
      if (!biggest || loss > biggest.loss) biggest = { ply: i + 1, loss };
    });
    const n = sideAnnots.length;
    // accuracy: rough mapping from counts
    const acc = Math.max(0, Math.min(1,
      (counts.best + counts.ok * 0.85 + counts.inaccuracy * 0.65 + counts.mistake * 0.35 + counts.blunder * 0.10) / Math.max(1, n)
    ));
    return { counts, totalLoss, biggest, n, accuracy: acc };
  }, [annotations, side, analyzed]);

  // current ply analysis
  const curAnnot = currentPly > 0 && annotations[currentPly - 1] ? annotations[currentPly - 1] : null;
  const curPos = positions[currentPly] || positions[0];
  const lastMove = curPos && curPos.move ? curPos.move : null;

  return (
    <div className="root">
      <ShellTopbar current="annotate" />
      <div className="page">
        <div className="page-header">
          <div className="page-eyebrow">tool · annotator</div>
          <h1 className="page-title">PGN annotator</h1>
          <p className="page-sub">Drop in a game — every ply runs through the model. Find your blunders, see what the bot would have played instead.</p>
        </div>

        {!analyzed ? (
          <Intake
            tab={intakeTab} setTab={setIntakeTab}
            pgnText={pgnText} setPgnText={setPgnText}
            side={side} setSide={setSide}
            onAnalyze={runAnalysis}
          />
        ) : (
          <>
            <VTrajectory
              annotations={annotations}
              positions={positions}
              currentPly={currentPly}
              onJump={setCurrentPly}
              side={side === 'both' ? 'w' : side}
            />
            <div className="ann-layout">
              <div className="ann-board-wrap">
                <div className="ann-board">
                  <MiniBoard fen={curPos.fen} lastMove={lastMove} pieceSize={48} />
                </div>
                <Scrubber currentPly={currentPly} total={positions.length} annotations={annotations} onSet={setCurrentPly} side={side} />
                <GameHeader headers={headers} positions={positions} onReset={() => setAnalyzed(false)} />
              </div>

              <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
                <PlyAnalysisCard currentPly={currentPly} annot={curAnnot} side={side} />
                <AltMovesCard annot={curAnnot} positions={positions} currentPly={currentPly} />
                <AggregateStatsCard stats={stats} onJump={setCurrentPly} side={side} />
                <AnnotatedMoveList annotations={annotations} currentPly={currentPly} onJump={setCurrentPly} side={side} />
              </div>
            </div>
          </>
        )}
      </div>
    </div>
  );
}

// ── Intake card ───────────────────────────────────────────────────────────
function Intake({ tab, setTab, pgnText, setPgnText, side, setSide, onAnalyze }) {
  return (
    <div className="ann-intake">
      <div className="card">
        <div className="intake-tabs">
          <button className={`intake-tab ${tab === 'paste' ? 'active' : ''}`} onClick={() => setTab('paste')}>paste pgn</button>
          <button className={`intake-tab ${tab === 'upload' ? 'active' : ''}`} onClick={() => setTab('upload')}>upload .pgn</button>
          <button className={`intake-tab ${tab === 'url' ? 'active' : ''}`} onClick={() => setTab('url')}>lichess url</button>
        </div>
        {tab === 'paste' && (
          <textarea className="textarea" value={pgnText} onChange={e => setPgnText(e.target.value)} spellCheck={false} />
        )}
        {tab === 'upload' && (
          <div style={{ padding: 24, border: '2px dashed var(--border-strong)', borderRadius: 6, textAlign: 'center', fontFamily: 'var(--font-mono)', fontSize: 13, color: 'var(--muted)' }}>
            drag a .pgn file here, or click to browse
          </div>
        )}
        {tab === 'url' && (
          <input className="input" placeholder="https://lichess.org/abc123" />
        )}
        <div className="sample-row">
          <span style={{ fontFamily: 'var(--font-mono)', fontSize: 11, color: 'var(--muted)', alignSelf: 'center' }}>or try:</span>
          <button className="sample-btn">italian, 30 mvs</button>
          <button className="sample-btn">caro-kann blunder</button>
          <button className="sample-btn">opera game (morphy)</button>
        </div>
      </div>

      <div className="card">
        <div className="section-label">annotate</div>
        <div className="as-row">
          <button className={`as-pill ${side === 'w' ? 'on' : ''}`} onClick={() => setSide('w')}>white</button>
          <button className={`as-pill ${side === 'b' ? 'on' : ''}`} onClick={() => setSide('b')}>black</button>
          <button className={`as-pill ${side === 'both' ? 'on' : ''}`} onClick={() => setSide('both')}>both</button>
        </div>
        <div style={{ flex: 1 }}></div>
        <div style={{ fontFamily: 'var(--font-mono)', fontSize: 11, color: 'var(--muted)', lineHeight: 1.5 }}>
          The bot evaluates every ply, compares your move to its top choice, and flags inaccuracies, mistakes, and blunders by V loss.
        </div>
        <button className="btn btn-primary" onClick={onAnalyze} style={{ marginTop: 'auto' }}>analyze →</button>
      </div>
    </div>
  );
}

// ── Scrubber ──────────────────────────────────────────────────────────────
function Scrubber({ currentPly, total, annotations, onSet, side }) {
  const trackRef = useRef(null);
  const onTrackClick = (e) => {
    const r = trackRef.current.getBoundingClientRect();
    const f = (e.clientX - r.left) / r.width;
    onSet(Math.max(0, Math.min(total - 1, Math.round(f * (total - 1)))));
  };
  const fullMove = currentPly === 0 ? 0 : Math.ceil(currentPly / 2);
  const fmoves = Math.ceil((total - 1) / 2);
  return (
    <div className="scrubber">
      <button className="scrub-btn" onClick={() => onSet(0)}>‹‹</button>
      <button className="scrub-btn" onClick={() => onSet(Math.max(0, currentPly - 1))}>‹</button>
      <div ref={trackRef} className="scrub-track" onClick={onTrackClick}>
        <div className="scrub-fill" style={{ width: `${(currentPly / Math.max(1, total - 1)) * 100}%` }} />
        {annotations.map((a, i) => {
          const myMove = side === 'both' || (i % 2 === 0 ? side === 'w' : side === 'b');
          if (!myMove) return null;
          if (a.verdict !== 'blunder' && a.verdict !== 'mistake') return null;
          const left = `${((i + 1) / Math.max(1, total - 1)) * 100}%`;
          return <div key={i} className={a.verdict === 'blunder' ? 'scrub-blunder' : 'scrub-mistake'} style={{ left }} />;
        })}
        <div className="scrub-handle" style={{ left: `${(currentPly / Math.max(1, total - 1)) * 100}%` }} />
      </div>
      <div className="scrub-pos">{fullMove} / {fmoves}</div>
      <button className="scrub-btn" onClick={() => onSet(Math.min(total - 1, currentPly + 1))}>›</button>
      <button className="scrub-btn" onClick={() => onSet(total - 1)}>››</button>
    </div>
  );
}

// ── Game header card (above scrubber, under board) ────────────────────────
function GameHeader({ headers, positions, onReset }) {
  return (
    <div className="card" style={{ width: '100%', padding: '14px 16px' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', gap: 16, alignItems: 'baseline' }}>
        <div>
          <div style={{ fontFamily: 'var(--font-mono)', fontSize: 11, color: 'var(--muted)' }}>{headers.Event || '—'} · {headers.Date || ''}</div>
          <div style={{ marginTop: 4, fontSize: 14, fontWeight: 500, letterSpacing: '-0.015em' }}>
            <span>{headers.White || 'white'}</span>
            <span style={{ color: 'var(--muted)', margin: '0 8px' }}>vs</span>
            <span>{headers.Black || 'black'}</span>
            <span className="pill" style={{ marginLeft: 10, verticalAlign: 'middle' }}>{headers.Result || '*'}</span>
          </div>
        </div>
        <button className="btn-ghost btn btn-sm" onClick={onReset}>← change pgn</button>
      </div>
    </div>
  );
}

// ── Current ply analysis card ─────────────────────────────────────────────
function PlyAnalysisCard({ currentPly, annot, side }) {
  if (!annot) {
    return (
      <div className="card ply-card">
        <div style={{ fontFamily: 'var(--font-mono)', fontSize: 12, color: 'var(--muted)', textAlign: 'center', padding: '12px 0' }}>
          → step forward to start
        </div>
      </div>
    );
  }
  const moveColor = (currentPly - 1) % 2 === 0 ? 'w' : 'b';
  const sign = moveColor === 'w' ? 1 : -1;
  const loss = (annot.best_v - annot.v) * sign;
  const showLoss = annot.verdict !== 'best' && annot.verdict !== 'ok';
  const v = annot.v;
  const bv = annot.best_v;
  const info = VERDICT_INFO[annot.verdict] || VERDICT_INFO.ok;

  return (
    <div className="card ply-card">
      <div className="section-label" style={{ marginBottom: 14 }}>
        ply {currentPly} · {moveColor === 'w' ? 'white' : 'black'} to move
      </div>
      <div className="move-grid">
        <span className="move-label">played</span>
        <span className={`move-san is-your`}>{ANNOT_SAN_AT(currentPly)}</span>
        <span className={`move-v ${v >= 0 ? 'pos' : 'neg'}`}>{fmtV(v)}</span>

        <span className="move-label">best</span>
        <span className={`move-san is-bot`}>{annot.best}</span>
        <span className={`move-v ${bv >= 0 ? 'pos' : 'neg'}`}>{fmtV(bv)}</span>
      </div>
      <div className="verdict-row">
        <span className={`verdict ${info.tone}`}>{info.glyph} {info.label}</span>
        {showLoss && (
          <span className="delta">V loss: {loss.toFixed(2)}</span>
        )}
      </div>
      <div style={{ fontFamily: 'var(--font-mono)', fontSize: 12, color: 'var(--ink-2)', lineHeight: 1.55, marginTop: 12 }}>
        {info.sentence}
      </div>
    </div>
  );
}

// helper that needs access to positions  — patch in via a closure-like pattern.
let _positions = [];
function ANNOT_SAN_AT(ply) {
  return _positions[ply] && _positions[ply].san ? _positions[ply].san : '–';
}

// ── Top alternatives panel ────────────────────────────────────────────────
function AltMovesCard({ annot, positions, currentPly }) {
  _positions = positions; // expose for sibling helper
  if (!annot) return null;
  const playedSan = positions[currentPly] && positions[currentPly].san;
  return (
    <div className="panel">
      <div className="panel-head">
        <span className="panel-title">top alternatives</span>
        <span className="panel-sub">policy + V_after</span>
      </div>
      <div className="panel-body">
        {annot.top.map((t, i) => {
          const [san, v, prob] = t;
          const isPlayed = san === playedSan;
          const isBest = i === 0;
          return (
            <div key={i} className={`alt-row ${isPlayed ? 'is-played' : ''} ${isBest ? 'is-best' : ''}`}>
              <span className="alt-rank">{i + 1}</span>
              <span className="alt-san">{san}</span>
              <span className="bar-cell"><span className="bar-cell-fill" style={{ width: `${Math.max(4, prob * 100)}%`, background: isBest ? 'var(--accent)' : 'var(--ink-2)' }} /></span>
              <span className={`alt-v ${v >= 0 ? 'pos' : 'neg'}`}>{fmtV(v)}</span>
            </div>
          );
        })}
      </div>
    </div>
  );
}

// ── Aggregate stats card ─────────────────────────────────────────────────
function AggregateStatsCard({ stats, onJump, side }) {
  if (!stats) return null;
  const { counts, accuracy, biggest, totalLoss, n } = stats;
  return (
    <div className="panel">
      <div className="panel-head">
        <span className="panel-title">summary</span>
        <span className="panel-sub">{n} {side === 'both' ? 'plies' : `${side === 'w' ? 'white' : 'black'} plies`}</span>
      </div>
      <div className="stat-grid">
        <div className="stat">
          <div className="stat-label">accuracy</div>
          <div className="stat-value">{(accuracy * 100).toFixed(0)}<span style={{ fontSize: 13, color: 'var(--muted)', fontFamily: 'var(--font-mono)' }}>%</span></div>
          <div className="acc-meter"><div className="acc-meter-fill" style={{ width: `${accuracy * 100}%` }} /></div>
        </div>
        <div className="stat">
          <div className="stat-label">V loss total</div>
          <div className="stat-value mono">{totalLoss.toFixed(2)}</div>
          <div className="stat-delta">avg {(totalLoss / Math.max(1, n)).toFixed(3)}/ply</div>
        </div>
        <div className="stat">
          <div className="stat-label">inaccuracies</div>
          <div className="stat-value mono" style={{ color: 'var(--warn)' }}>{counts.inaccuracy || 0}</div>
        </div>
        <div className="stat">
          <div className="stat-label">mistakes</div>
          <div className="stat-value mono" style={{ color: 'var(--neg)' }}>{counts.mistake || 0}</div>
        </div>
        <div className="stat" style={{ gridColumn: 'span 2' }}>
          <div className="stat-label">blunders</div>
          <div style={{ display: 'flex', alignItems: 'baseline', gap: 10 }}>
            <span className="stat-value mono" style={{ color: 'var(--neg)' }}>{counts.blunder || 0}</span>
            {biggest && (
              <button className="btn btn-ghost btn-sm" style={{ marginLeft: 'auto' }} onClick={() => onJump(biggest.ply)}>
                → biggest: ply {biggest.ply} (Δ{biggest.loss.toFixed(2)})
              </button>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

// ── Annotated move list ──────────────────────────────────────────────────
function AnnotatedMoveList({ annotations, currentPly, onJump, side }) {
  // Pair up plies (1.= white+black)
  const pairs = [];
  for (let i = 0; i < annotations.length; i += 2) {
    pairs.push([
      annotations[i] ? { annot: annotations[i], ply: i + 1, san: ANNOT_SAN_AT(i + 1) } : null,
      annotations[i+1] ? { annot: annotations[i+1], ply: i + 2, san: ANNOT_SAN_AT(i + 2) } : null,
    ]);
  }
  const scrollRef = useRef();
  useEffect(() => {
    if (!scrollRef.current) return;
    const el = scrollRef.current.querySelector('.is-current');
    if (el) el.scrollIntoView({ block: 'center' });
  }, [currentPly]);

  const renderCell = (cell) => {
    if (!cell) return <span className="ann-mlcell" />;
    const verdict = cell.annot.verdict;
    const v = VERDICT_INFO[verdict];
    const myMove = side === 'both' || ((cell.ply - 1) % 2 === 0 ? side === 'w' : side === 'b');
    const showMarker = myMove && verdict !== 'best' && verdict !== 'ok';
    return (
      <span className="ann-mlcell" onClick={() => onJump(cell.ply)} style={!myMove ? { opacity: 0.5 } : {}}>
        {cell.san}
        {showMarker && <span className={`marker ${v.tone}`}>{v.glyph}</span>}
      </span>
    );
  };

  return (
    <div className="panel">
      <div className="panel-head">
        <span className="panel-title">annotated moves</span>
        <span className="panel-sub">click any ply</span>
      </div>
      <div className="ann-moves" ref={scrollRef}>
        {pairs.map(([w, b], i) => (
          <div key={i} className={`ann-mlrow ${(w && w.ply === currentPly) || (b && b.ply === currentPly) ? 'is-current' : ''}`}>
            <span style={{ color: 'var(--muted)', fontSize: 11 }}>{i + 1}.</span>
            {renderCell(w)}
            {renderCell(b)}
          </div>
        ))}
      </div>
    </div>
  );
}

ReactDOM.createRoot(document.getElementById('root')).render(<App />);
