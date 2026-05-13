// pieces.jsx — Bottisme chess pieces.
// Uses Unicode chess glyphs (U+265A–F, filled silhouettes) for both colours,
// with CSS coloring to differentiate white (light fill + ink stroke) from
// black (ink fill). Unicode renders reliably across browsers from system
// fonts (Apple Symbols / Segoe UI Symbol / Noto Sans Symbols 2).

const GLYPHS = {
  k: '\u265A', q: '\u265B', r: '\u265C',
  b: '\u265D', n: '\u265E', p: '\u265F',
};

function Piece({ type, color, size = 64, dragging = false }) {
  const glyph = GLYPHS[type];
  if (!glyph) return null;
  return (
    <span
      className={`piece piece-${color}${dragging ? ' piece-drag' : ''}`}
      style={{ fontSize: size + 'px', lineHeight: 1 }}
      aria-label={`${color === 'w' ? 'White' : 'Black'} ${type}`}
    >
      {glyph}
    </span>
  );
}

window.Piece = Piece;
