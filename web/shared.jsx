// shared.jsx — components used across multiple Bottisme app pages.
// Loaded after pieces.jsx (window.Piece) and engine.jsx (window.BotEngine).

// ── App shell topbar ──────────────────────────────────────────────────────
// Drop-in topbar with brand wordmark, breadcrumb-style nav, and a "tools"
// dropdown that lists every app in the suite.
const TOOLS = [
  { id: 'play',      label: 'Play',                file: 'Bottisme.html',  category: 'core' },
  { id: 'annotate',  label: 'PGN Annotator',       file: 'Annotate.html',  category: 'analyse' },
  { id: 'analyze',   label: 'Position Analyzer',   file: 'Analyze.html',   category: 'analyse' },
];

function ShellTopbar({ current, breadcrumb }) {
  const [open, setOpen] = React.useState(false);
  const ref = React.useRef(null);
  React.useEffect(() => {
    const onDoc = (e) => { if (ref.current && !ref.current.contains(e.target)) setOpen(false); };
    document.addEventListener('mousedown', onDoc);
    return () => document.removeEventListener('mousedown', onDoc);
  }, []);
  const curr = TOOLS.find(t => t.id === current);

  return (
    <div className="shell-topbar">
      <div className="shell-nav">
        <a href="Hub.html" className="brand brand-small">
          <span className="brand-mark">bottisme</span><span className="brand-dot">.</span>
        </a>
        <span className="crumb-sep">/</span>
        {breadcrumb ? (
          <>
            <a href="Hub.html">tools</a>
            <span className="crumb-sep">/</span>
            <span className="crumb-current">{breadcrumb}</span>
          </>
        ) : (
          <span className="crumb-current">{curr ? curr.label.toLowerCase() : ''}</span>
        )}
      </div>
      <div className="shell-tools-wrap" ref={ref}>
        <button className="shell-tools-trigger" onClick={() => setOpen(o => !o)}>
          tools <span style={{ opacity: 0.5 }}>{open ? '▴' : '▾'}</span>
        </button>
        {open && (
          <div className="shell-tools-menu">
            {['core', 'analyse', 'deploy', 'research'].map(cat => (
              <div key={cat}>
                <div className="tools-cat">{cat}</div>
                {TOOLS.filter(t => t.category === cat).map(t => (
                  <a key={t.id} href={t.file} className={`tools-item ${t.id === current ? 'is-current' : ''}`}>
                    {t.label}
                    {t.id === current && <span className="tools-current-dot" />}
                  </a>
                ))}
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

// ── MiniBoard ─────────────────────────────────────────────────────────────
// Lightweight board for previews / puzzles / inline mentions. Accepts a FEN
// and optional move highlights, lastMove, hint squares.
function MiniBoard({ fen, orientation = 'w', lastMove, highlight, pieceSize, palette = 'classic' }) {
  const chess = new Chess();
  if (fen) chess.load(fen);
  const board = chess.board();
  const rows = orientation === 'w' ? [0,1,2,3,4,5,6,7] : [7,6,5,4,3,2,1,0];
  const cols = orientation === 'w' ? [0,1,2,3,4,5,6,7] : [7,6,5,4,3,2,1,0];
  const F = 'abcdefgh';
  const sqName = (f, r) => F[f] + (r + 1);
  const hl = new Set(highlight || []);
  const ps = pieceSize || 28;
  return (
    <div className={`mini-board board-${palette}`}>
      {rows.map(r => (
        <div className="board-row" key={r}>
          {cols.map(c => {
            const piece = board[r][c];
            const file = c, rank = 7 - r;
            const name = sqName(file, rank);
            const isLight = (file + rank) % 2 === 1;
            const isLast = lastMove && (lastMove.from === name || lastMove.to === name);
            const isHl = hl.has(name);
            const cls = ['sq', isLight ? 'sq-light' : 'sq-dark', isLast && 'sq-last', isHl && 'sq-hint'].filter(Boolean).join(' ');
            return (
              <div key={name} className={cls}>
                {piece && <Piece type={piece.type} color={piece.color} size={ps} />}
              </div>
            );
          })}
        </div>
      ))}
    </div>
  );
}

// ── Tool card (used on Hub) ──────────────────────────────────────────────
function ToolCard({ tool, status, description, accent }) {
  return (
    <a href={tool.file} className={`tool-card${accent ? ' tool-card-accent' : ''}`}>
      <div className="tool-card-head">
        <span className="tool-card-id">{tool.id}</span>
        {status && <span className={`pill ${status.tone || ''}`}><span className="pill-dot" />{status.label}</span>}
      </div>
      <div className="tool-card-title">{tool.label}</div>
      <div className="tool-card-desc">{description}</div>
      <div className="tool-card-arrow">→</div>
    </a>
  );
}

// ── Formatting helpers ───────────────────────────────────────────────────
function fmtV(v) {
  if (v === null || v === undefined || isNaN(v)) return '–';
  return (v >= 0 ? '+' : '−') + Math.abs(v).toFixed(2);
}
function fmtPct(p) { return Math.round(p * 100) + '%'; }
function relTime(d) {
  const diff = (Date.now() - d) / 1000;
  if (diff < 60) return Math.floor(diff) + 's ago';
  if (diff < 3600) return Math.floor(diff / 60) + 'm ago';
  if (diff < 86400) return Math.floor(diff / 3600) + 'h ago';
  return Math.floor(diff / 86400) + 'd ago';
}

Object.assign(window, {
  ShellTopbar, MiniBoard, ToolCard, TOOLS, fmtV, fmtPct, relTime,
});
