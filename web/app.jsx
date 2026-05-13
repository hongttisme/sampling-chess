// app.jsx — Bottisme web play UI.

const { useState, useEffect, useRef, useCallback, useMemo } = React;

// ── tweak defaults ────────────────────────────────────────────────────────
const TWEAK_DEFAULTS = /*EDITMODE-BEGIN*/{
  "theme": "paper",
  "boardPalette": "classic",
  "density": "cozy",
  "showHints": true,
  "showEvalBar": true,
  "evalFromYourSide": true,
  "highlightLastMove": true,
  "coordinates": true
}/*EDITMODE-END*/;

// ── helpers ───────────────────────────────────────────────────────────────
const FILES = ['a','b','c','d','e','f','g','h'];
const RANKS = ['1','2','3','4','5','6','7','8'];
const sqName = (file, rank) => FILES[file] + RANKS[rank];

function formatV(v) {
  if (v === null || v === undefined || isNaN(v)) return '–';
  const sign = v >= 0 ? '+' : '−';
  return sign + Math.abs(v).toFixed(2);
}

function moveSan(move) {
  // Use chess.js's .san already produced when moving verbose.
  return move && move.san ? move.san : '';
}

function copy(text) {
  try { navigator.clipboard.writeText(text); } catch {}
}

// ── Board component ───────────────────────────────────────────────────────
function Board({
  fen, orientation, selected, legalTargets, lastMove, onSquareClick,
  inCheckSquare, hintSquares, boardPalette, coordinates, highlightLastMove, pieceSize,
}) {
  const chessRef = useRef(new Chess(fen));
  chessRef.current.load(fen);
  const board = chessRef.current.board(); // [rank8..rank1][fileA..fileH]

  // Flip for orientation
  const rows = orientation === 'w' ? [0,1,2,3,4,5,6,7] : [7,6,5,4,3,2,1,0];
  const cols = orientation === 'w' ? [0,1,2,3,4,5,6,7] : [7,6,5,4,3,2,1,0];

  return (
    <div className={`board board-${boardPalette}`} role="grid" aria-label="Chess board">
      {rows.map(r => (
        <div className="board-row" key={r} role="row">
          {cols.map(c => {
            const piece = board[r][c]; // {type, color} or null
            const file = c, rank = 7 - r;
            const name = sqName(file, rank);
            const isLight = (file + rank) % 2 === 1;
            const isSelected = selected === name;
            const isLegal = legalTargets && legalTargets.has(name);
            const isCapture = isLegal && piece;
            const isLast = highlightLastMove && lastMove &&
                           (lastMove.from === name || lastMove.to === name);
            const isCheck = inCheckSquare === name;
            const isHint = hintSquares && hintSquares.has(name);

            const showFile = coordinates && (orientation === 'w' ? r === 7 : r === 0);
            const showRank = coordinates && (orientation === 'w' ? c === 0 : c === 7);

            const cls = [
              'sq',
              isLight ? 'sq-light' : 'sq-dark',
              isSelected && 'sq-selected',
              isLast && 'sq-last',
              isCheck && 'sq-check',
              isHint && 'sq-hint',
            ].filter(Boolean).join(' ');

            return (
              <div
                key={name}
                role="gridcell"
                className={cls}
                onClick={() => onSquareClick(name)}
                data-square={name}
              >
                {showFile && <span className="coord coord-file">{FILES[file]}</span>}
                {showRank && <span className="coord coord-rank">{RANKS[rank]}</span>}
                {piece && <Piece type={piece.type} color={piece.color} size={pieceSize} />}
                {isLegal && !isCapture && <span className="legal-dot" />}
                {isCapture && <span className="legal-ring" />}
              </div>
            );
          })}
        </div>
      ))}
    </div>
  );
}

// ── Eval bar (vertical) ───────────────────────────────────────────────────
function EvalBar({ v, playerColor, evalFromYourSide }) {
  // v is white's perspective in [-1, 1].
  // Convert to "your share" (0..1) — your side fills from the bottom.
  const yourShare = playerColor === 'w' ? (v + 1) / 2 : (1 - v) / 2;
  const yourPct = Math.max(0.04, Math.min(0.96, yourShare)) * 100;

  const youIsWhite = playerColor === 'w';
  const displayedV = evalFromYourSide ? (youIsWhite ? v : -v) : v;

  return (
    <div className={`eval-bar eval-you-${playerColor}`} title={`Position eval: ${formatV(v)} (white's perspective)`}>
      <div className="eval-fill-top"    style={{ height: `${100 - yourPct}%` }} />
      <div className="eval-fill-bottom" style={{ height: `${yourPct}%` }} />
      <div className="eval-midline" />
      <div className="eval-label">{formatV(displayedV)}</div>
    </div>
  );
}

// ── Top moves panel ───────────────────────────────────────────────────────
function TopMovesPanel({ analysis, isYourTurn, thinking, playerColor, evalFromYourSide, onHover, onLeave }) {
  const moves = analysis ? analysis.topMoves : [];
  return (
    <div className="panel panel-topk">
      <div className="panel-head">
        <span className="panel-title">policy</span>
        <span className="panel-sub">
          {thinking ? 'computing…' : (isYourTurn ? 'recommended for you' : "bot's options")}
        </span>
      </div>
      <div className="panel-body">
        {moves.length === 0 ? (
          <div className="empty">—</div>
        ) : moves.map((m, i) => {
          const v = evalFromYourSide
            ? (playerColor === 'w' ? m.vAfter : -m.vAfter)
            : m.vAfter;
          const pct = Math.round(m.prob * 100);
          return (
            <div
              key={i}
              className={`move-row${i === 0 ? ' move-row-best' : ''}`}
              onMouseEnter={() => onHover && onHover(m.move)}
              onMouseLeave={() => onLeave && onLeave()}
            >
              <span className="rank">{i + 1}</span>
              <span className="san">{moveSan(m.move)}</span>
              <span className="bar-cell"><span className="bar-cell-fill" style={{ width: `${Math.max(2, pct)}%` }} /></span>
              <span className="pct">{pct}%</span>
              <span className={`vAfter ${v >= 0 ? 'pos' : 'neg'}`}>{formatV(v)}</span>
            </div>
          );
        })}
      </div>
    </div>
  );
}

// ── Move list panel ───────────────────────────────────────────────────────
function MoveListPanel({ history, playerColor, evalFromYourSide }) {
  // history: list of plies: { san, color, vAfter, mistake }
  const rows = [];
  for (let i = 0; i < history.length; i += 2) {
    rows.push([history[i], history[i + 1]]);
  }
  const scrollRef = useRef(null);
  useEffect(() => {
    if (scrollRef.current) scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
  }, [history.length]);

  return (
    <div className="panel panel-moves">
      <div className="panel-head">
        <span className="panel-title">moves</span>
        <span className="panel-sub">{history.length} ply</span>
      </div>
      <div className="panel-body move-list" ref={scrollRef}>
        {rows.length === 0 ? (
          <div className="empty">no moves yet</div>
        ) : rows.map((pair, i) => (
          <div className="ml-row" key={i}>
            <span className="ml-num">{i + 1}.</span>
            <span className={`ml-cell${pair[0]?.mistake ? ' mistake' : ''}`}>
              {pair[0] ? pair[0].san : ''}
            </span>
            <span className={`ml-cell${pair[1]?.mistake ? ' mistake' : ''}`}>
              {pair[1] ? pair[1].san : ''}
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}

// ── Promotion modal ───────────────────────────────────────────────────────
function PromotionModal({ color, onPick, onCancel }) {
  const opts = ['q', 'r', 'b', 'n'];
  return (
    <div className="modal-overlay" onClick={onCancel}>
      <div className="modal promotion-modal" onClick={e => e.stopPropagation()}>
        <div className="modal-title">promote pawn</div>
        <div className="promo-row">
          {opts.map(t => (
            <button key={t} className="promo-btn" onClick={() => onPick(t)}>
              <Piece type={t} color={color} size={56} />
            </button>
          ))}
        </div>
      </div>
    </div>
  );
}

// ── V-trend sparkline ─────────────────────────────────────────────────────
function VTrendPanel({ history, playerColor, evalFromYourSide }) {
  // history is ordered plies. vAfter is from white's perspective in [-1,1].
  // We'll plot vAfter (or negated for black) at each ply.
  const vs = history.map(h => {
    const v = h.vAfter ?? 0;
    return (evalFromYourSide && playerColor === 'b') ? -v : v;
  });
  // pad start with 0
  const pts = [0, ...vs];

  const W = 320, H = 64, PAD = 6;
  const innerW = W - PAD * 2;
  const innerH = H - PAD * 2;
  const n = pts.length;
  // Auto-scale Y to give early-game small values visible amplitude. Floor at ±0.3.
  const maxAbs = Math.max(0.3, ...pts.map(Math.abs)) * 1.15;
  const x = (i) => PAD + (n === 1 ? innerW / 2 : (i / Math.max(1, n - 1)) * innerW);
  const y = (v) => {
    const norm = Math.max(-1, Math.min(1, v / maxAbs)); // -1..1
    return PAD + innerH * (1 - (norm + 1) / 2);
  };

  const path = pts.map((v, i) => `${i === 0 ? 'M' : 'L'} ${x(i).toFixed(1)} ${y(v).toFixed(1)}`).join(' ');
  const areaPath = `${path} L ${x(n-1).toFixed(1)} ${y(0).toFixed(1)} L ${x(0).toFixed(1)} ${y(0).toFixed(1)} Z`;
  const last = pts[pts.length - 1] ?? 0;
  const trend = pts.length >= 2 ? pts[pts.length - 1] - pts[pts.length - 2] : 0;

  return (
    <div className="panel panel-trend">
      <div className="panel-head">
        <span className="panel-title">V trend</span>
        <span className="panel-sub">{evalFromYourSide ? 'your side' : "white's side"}</span>
      </div>
      <div className="panel-body trend-body">
        <svg width="100%" viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="none" className="trend-svg">
          <defs>
            <linearGradient id="trendgrad" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor="var(--pos)" stopOpacity="0.25" />
              <stop offset="50%" stopColor="var(--pos)" stopOpacity="0.05" />
              <stop offset="100%" stopColor="var(--pos)" stopOpacity="0" />
            </linearGradient>
          </defs>
          {/* midline */}
          <line x1={PAD} y1={y(0)} x2={W - PAD} y2={y(0)} stroke="var(--border-strong)" strokeWidth="1" strokeDasharray="2 3" />
          {/* scale label */}
          <text x={W - PAD - 2} y={PAD + 8} fontFamily="var(--font-mono)" fontSize="8" fill="var(--muted)" textAnchor="end">±{maxAbs.toFixed(1)}</text>
          {n > 1 && <path d={areaPath} fill="url(#trendgrad)" />}
          {n > 1 && <path d={path} stroke="var(--ink)" strokeWidth="1.4" fill="none" strokeLinejoin="round" strokeLinecap="round" />}
          {n > 0 && <circle cx={x(n-1)} cy={y(last)} r="2.6" fill="var(--accent)" />}
        </svg>
        <div className="trend-stat">
          <span className="ts-label">now</span>
          <span className={`ts-val ${last >= 0 ? 'pos' : 'neg'}`}>{formatV(last)}</span>
          <span className="ts-sep">·</span>
          <span className="ts-label">Δ</span>
          <span className={`ts-val ${trend >= 0 ? 'pos' : 'neg'}`}>{formatV(trend)}</span>
        </div>
      </div>
    </div>
  );
}

window.BottismeUI = { Board, EvalBar, TopMovesPanel, MoveListPanel, PromotionModal, VTrendPanel };
window.useBottismeTweakDefaults = () => TWEAK_DEFAULTS;
