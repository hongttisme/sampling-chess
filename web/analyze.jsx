// analyze.jsx — Bottisme Position Analyzer (App 5).
// Uses window.BotEngine (engine.jsx) for V + policy and window.MiniBoard
// (shared.jsx) for the board render. Click a row in the top-10 list to
// see the "what if I play X" comparison, including the bot's reply.

const { useState, useMemo, useCallback, useEffect } = React;

const STARTPOS_FEN = 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1';

const PRESETS = [
  { label: 'Startpos',         fen: STARTPOS_FEN },
  { label: 'Italian opening',  fen: 'r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3' },
  { label: 'Sicilian Najdorf', fen: 'rnbqkb1r/1p2pppp/p2p1n2/8/3NP3/2N5/PPP2PPP/R1BQKB1R w KQkq - 0 6' },
  { label: 'Endgame K+R vs K', fen: '4k3/8/8/8/8/8/8/4K2R w K - 0 1' },
  { label: 'Mate in 1 (W)',    fen: '7k/5Q2/6K1/8/8/8/8/8 w - - 0 1' },
];

// V from white POV → human description.
function vDesc(v, sideToMove) {
  const a = Math.abs(v);
  let strength;
  if (a < 0.05)      strength = 'roughly balanced';
  else if (a < 0.20) strength = 'slight edge';
  else if (a < 0.45) strength = 'clear advantage';
  else if (a < 0.80) strength = 'winning';
  else               strength = 'decisive';
  if (a < 0.05) return strength;
  const who = v > 0 ? 'white' : 'black';
  return `${who} ${strength}`;
}

// Compute a 5-ply principal variation by greedy unrolling of analyze().
function computePV(fen, maxPly = 5) {
  const chess = new Chess();
  chess.load(fen);
  const moves = [];
  for (let i = 0; i < maxPly; i++) {
    if (chess.game_over()) break;
    const a = window.BotEngine.analyze(chess, 0);
    if (!a.scored.length) break;
    const top = a.scored[0];
    moves.push({
      san: top.move.san,
      from: top.move.from,
      to: top.move.to,
      vAfter: top.vAfter,
      color: top.move.color,
    });
    chess.move(top.move);
  }
  return moves;
}

function FenRow({ fen, setFen, onLoad, onReset }) {
  return (
    <div className="fen-row">
      <input
        className="input"
        type="text"
        value={fen}
        onChange={e => setFen(e.target.value)}
        spellCheck={false}
      />
      <button className="btn btn-primary" onClick={onLoad}>Load</button>
      <button className="btn" onClick={onReset}>Startpos</button>
      <button className="btn" onClick={() => navigator.clipboard?.writeText(fen)}>Copy</button>
    </div>
  );
}

function PresetRow({ onPick }) {
  return (
    <div className="preset-row">
      <span className="muted-label" style={{ marginRight: 4 }}>presets:</span>
      {PRESETS.map(p => (
        <button key={p.label} className="btn btn-small" onClick={() => onPick(p.fen)}>
          {p.label}
        </button>
      ))}
    </div>
  );
}

function VDisplay({ v, sideToMove, ply }) {
  const cls = v > 0.05 ? 'pos' : v < -0.05 ? 'neg' : '';
  return (
    <div className="v-display">
      <div className={`v-big ${cls}`}>{window.fmtV(v)}</div>
      <div className="v-desc">{vDesc(v, sideToMove)}</div>
      <div className="v-trail">
        <span className="turn-pill">
          <span className="pill-dot" />
          {sideToMove === 'w' ? 'white to move' : 'black to move'}
        </span>
      </div>
    </div>
  );
}

function PVCard({ pv, sideToMove, onJump }) {
  if (!pv.length) return null;
  // Group plies into move pairs for SAN display "1. e4 e5  2. Nf3 ..."
  const startPly = sideToMove === 'w' ? 0 : 1;
  return (
    <div className="card pv-card">
      <div className="card-h">
        <div className="card-title">Principal variation</div>
        <div className="card-meta">{pv.length} plies, greedy</div>
      </div>
      <div className="pv-moves">
        {pv.map((m, i) => {
          const moveNum = Math.floor((startPly + i) / 2) + 1;
          const showNum = (startPly + i) % 2 === 0;
          return (
            <React.Fragment key={i}>
              {showNum && <span className="pv-move-num">{moveNum}.</span>}
              <span
                className={`pv-move ${i === 0 ? 'is-bot' : ''}`}
                onClick={() => onJump && onJump(i)}
                title={`V_after = ${window.fmtV(m.vAfter)}`}
              >
                {m.san}
              </span>
            </React.Fragment>
          );
        })}
      </div>
    </div>
  );
}

function CandidateList({ scored, selected, onSelect }) {
  if (!scored.length) {
    return (
      <div className="card">
        <div className="card-h"><div className="card-title">Top candidates</div></div>
        <div className="branch-empty">No legal moves (terminal position).</div>
      </div>
    );
  }
  const top = scored.slice(0, 10);
  const maxProb = Math.max(...top.map(s => s.prob), 0.01);
  return (
    <div className="card">
      <div className="card-h">
        <div className="card-title">Top candidates</div>
        <div className="card-meta">{scored.length} legal · top 10</div>
      </div>
      <div className="alt-list">
        {top.map((s, i) => {
          const isSel = selected && selected.move.san === s.move.san;
          const moverSign = s.move.color === 'w' ? 1 : -1;
          const vAfterMover = s.vAfter * moverSign;
          const vCls = vAfterMover > 0.05 ? 'pos' : vAfterMover < -0.05 ? 'neg' : '';
          return (
            <div
              key={s.move.san + i}
              className={`alt-row ${isSel ? 'is-selected' : ''}`}
              onClick={() => onSelect(s)}
            >
              <span className="alt-rank">#{i + 1}</span>
              <span className="alt-move">{s.move.san}</span>
              <div className="alt-bar">
                <div
                  className="alt-bar-fill"
                  style={{ width: (100 * s.prob / maxProb) + '%' }}
                />
              </div>
              <span className="alt-prob">{window.fmtPct(s.prob)}</span>
              <span className={`alt-v ${vCls}`}>V {window.fmtV(s.vAfter)}</span>
            </div>
          );
        })}
      </div>
    </div>
  );
}

function BranchCard({ rootV, selected, sideToMove, branchPV }) {
  if (!selected) {
    return (
      <div className="card branch-card">
        <div className="card-h">
          <div className="card-title">What if?</div>
          <div className="card-meta">click a candidate</div>
        </div>
        <div className="branch-empty">
          Select a move from the top candidates to see the resulting position
          and the bot's likely reply.
        </div>
      </div>
    );
  }
  const moverSign = selected.move.color === 'w' ? 1 : -1;
  const vAfterMover = selected.vAfter * moverSign;
  const rootVMover = rootV * moverSign;
  const delta = vAfterMover - rootVMover;
  const cls = vAfterMover > 0.05 ? 'pos' : vAfterMover < -0.05 ? 'neg' : '';
  const deltaStr = (delta >= 0 ? '+' : '') + delta.toFixed(2);
  return (
    <div className="card branch-card">
      <div className="card-h">
        <div className="card-title">What if {selected.move.san}?</div>
        <div className="card-meta">{window.fmtPct(selected.prob)} prob</div>
      </div>
      <div className="branch-comp">
        <span>after {selected.move.san}</span>
        <span className="arrow">→</span>
        <span className={`branch-v ${cls}`}>{window.fmtV(selected.vAfter)}</span>
        <span className="branch-delta">Δ {deltaStr} from root</span>
      </div>
      {branchPV.length > 0 && (
        <div className="pv-moves" style={{ marginTop: 12 }}>
          <span className="muted-label" style={{ marginRight: 6 }}>likely continuation:</span>
          {branchPV.map((m, i) => (
            <span
              key={i}
              className={`pv-move ${i === 0 ? 'is-bot' : ''}`}
              title={`V_after = ${window.fmtV(m.vAfter)}`}
            >
              {m.san}
            </span>
          ))}
        </div>
      )}
    </div>
  );
}

function Analyzer() {
  const [fen, setFen] = useState(STARTPOS_FEN);
  const [loadedFen, setLoadedFen] = useState(STARTPOS_FEN);
  const [selected, setSelected] = useState(null);
  const [error, setError] = useState(null);

  // Validate + parse loaded FEN.
  const chess = useMemo(() => {
    try {
      const c = new Chess();
      const ok = c.load(loadedFen);
      if (!ok) throw new Error('chess.js rejected the FEN');
      setError(null);
      return c;
    } catch (e) {
      setError(e.message || 'invalid FEN');
      const c = new Chess();
      return c;
    }
  }, [loadedFen]);

  // Run model analysis on the loaded FEN.
  const analysis = useMemo(() => {
    if (!chess) return { vCurrent: 0, scored: [], topMoves: [] };
    return window.BotEngine.analyze(chess, 0);
  }, [chess]);

  const pv = useMemo(() => computePV(chess.fen(), 5), [chess]);

  // PV branch starting AFTER the selected candidate move (4 plies for brevity).
  const branchPV = useMemo(() => {
    if (!selected) return [];
    const c = new Chess();
    c.load(chess.fen());
    c.move(selected.move);
    return computePV(c.fen(), 4);
  }, [selected, chess]);

  // Reset selection on FEN change.
  useEffect(() => { setSelected(null); }, [loadedFen]);

  const onLoad = () => setLoadedFen(fen.trim());
  const onReset = () => { setFen(STARTPOS_FEN); setLoadedFen(STARTPOS_FEN); };
  const onPickPreset = (preset) => { setFen(preset); setLoadedFen(preset); };

  // Highlight selected move's from/to squares on the board.
  const highlight = selected ? [selected.move.from, selected.move.to] : (
    pv.length ? [pv[0].from, pv[0].to] : []
  );
  const sideToMove = chess.turn();

  return (
    <div className="root">
      <ShellTopbar current="analyze" />
      <div className="page">
        <FenRow fen={fen} setFen={setFen} onLoad={onLoad} onReset={onReset} />
        <PresetRow onPick={onPickPreset} />
        {error && (
          <div className="card" style={{ padding: 12, marginBottom: 18, borderColor: 'var(--neg)' }}>
            <div className="muted-label" style={{ color: 'var(--neg)' }}>
              FEN error: {error}
            </div>
          </div>
        )}

        <div className="az-layout">
          <div className="az-board-wrap">
            <div className="az-board">
              <MiniBoard
                fen={chess.fen()}
                orientation={sideToMove}
                highlight={highlight}
                pieceSize={56}
                palette="classic"
              />
            </div>
            <VDisplay
              v={analysis.vCurrent}
              sideToMove={sideToMove}
              ply={chess.history().length}
            />
            <PVCard pv={pv} sideToMove={sideToMove} />
          </div>

          <div className="az-side">
            <CandidateList
              scored={analysis.scored}
              selected={selected}
              onSelect={setSelected}
            />
            <div style={{ height: 16 }} />
            <BranchCard
              rootV={analysis.vCurrent}
              selected={selected}
              sideToMove={sideToMove}
              branchPV={branchPV}
            />
          </div>
        </div>

        <div className="page-foot" style={{ marginTop: 24 }}>
          <span>bottisme · position analyzer</span>
          <span className="dot-sep">·</span>
          <a href="Hub.html">back to hub</a>
          <span className="dot-sep">·</span>
          <a href="Bottisme.html">play a game</a>
        </div>
      </div>
    </div>
  );
}

ReactDOM.createRoot(document.getElementById('root')).render(<Analyzer />);
