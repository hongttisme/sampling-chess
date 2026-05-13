// game.jsx — Bottisme game shell: welcome, setup, game, game-over states.

// formatV is needed by BotStatus. (Each Babel script gets its own scope, so
// we redeclare here rather than rely on main.jsx's copy.)
function formatV(v) {
  if (v === null || v === undefined || isNaN(v)) return '–';
  const sign = v >= 0 ? '+' : '−';
  return sign + Math.abs(v).toFixed(2);
}

// ── Welcome screen ────────────────────────────────────────────────────────
function Welcome({ onStart }) {
  return (
    <div className="welcome">
      <div className="welcome-inner">
        <div className="brand brand-large">
          <span className="brand-mark">bottisme<span className="brand-dot">.</span></span>
        </div>
        <div className="welcome-tag">
          a chess transformer that wants<br/>to play with you.
        </div>
        <div className="welcome-meta">
          <div className="meta-row">
            <span className="meta-label">model</span>
            <span className="meta-val">ChessTransformer · 8L · 6H</span>
          </div>
          <div className="meta-row">
            <span className="meta-label">trained</span>
            <span className="meta-val">SL on Lichess Elite ⤿ self-play</span>
          </div>
          <div className="meta-row">
            <span className="meta-label">latency</span>
            <span className="meta-val">~12 ms · CPU</span>
          </div>
        </div>
        <button className="play-btn" onClick={onStart}>
          play a game →
        </button>
        <div className="welcome-foot">
          <span>v0.4 · build a1b2c3</span>
          <span className="dot-sep">·</span>
          <a href="Hub.html">all tools</a>
          <span className="dot-sep">·</span>
          <a href="#" onClick={e => e.preventDefault()}>about</a>
          <span className="dot-sep">·</span>
          <a href="#" onClick={e => e.preventDefault()}>github</a>
        </div>
      </div>
    </div>
  );
}

// ── Sim-depth picker (search budget) ──────────────────────────────────────
// Lets the user pick how many MCTS simulations the bot runs per move.
// numSims === 0 means "no search" (raw policy net only).
// msPerSim is the per-sim wall-clock cost reported by the backend; while
// the real bot isn't connected we use a placeholder estimate.
function SimDepthPicker({ numSims, setNumSims, msPerSim }) {
  const presets = [
    { label: 'no search',    sims: 0   },
    { label: 'fast',         sims: 50  },
    { label: 'standard',     sims: 200 },
    { label: 'strong',       sims: 800 },
    { label: 'max',          sims: 1600 },
  ];
  const formatTime = (sims) => {
    if (sims === 0) return 'instant';
    const ms = sims * msPerSim;
    if (ms < 1000) return `~${Math.round(ms)}ms`;
    return `~${(ms / 1000).toFixed(ms < 5000 ? 1 : 0)}s`;
  };
  return (
    <div className="setup-section">
      <div className="setup-label">
        thinking depth
        <span className="setup-label-meta" style={{ marginLeft: 8, color: 'var(--muted)', fontWeight: 'normal', fontSize: '11px' }}>
          MCTS sims · est. {msPerSim.toFixed(1)} ms/sim on this device
        </span>
      </div>
      <div className="setup-options setup-modes">
        {presets.map(p => (
          <button
            key={p.label}
            className={`opt opt-mode ${numSims === p.sims ? 'opt-on' : ''}`}
            onClick={() => setNumSims(p.sims)}
          >
            <span className="opt-name">{p.label}</span>
            <span className="opt-sub">{p.sims === 0 ? 'policy net only' : `${p.sims} sims`}</span>
            <span className="opt-tag">{formatTime(p.sims)}</span>
          </button>
        ))}
      </div>
      <div style={{ marginTop: 8, fontSize: '11px', color: 'var(--muted)', fontFamily: 'var(--font-mono)' }}>
        custom: <input
          type="range" min="0" max="2000" step="50" value={numSims}
          onChange={e => setNumSims(parseInt(e.target.value))}
          style={{ width: 160, verticalAlign: 'middle', margin: '0 8px' }}
        />
        <span style={{ display: 'inline-block', minWidth: 60 }}>{numSims} sims</span>
        <span style={{ marginLeft: 8 }}>→ {formatTime(numSims)}</span>
      </div>
    </div>
  );
}

// ── Setup card ────────────────────────────────────────────────────────────
function Setup({ onStart, onBack }) {
  const [color, setColor] = React.useState('w');
  const [mode, setMode] = React.useState('greedy');
  // Default: standard search (200 sims). 0 = no search (net only).
  const [numSims, setNumSims] = React.useState(200);
  // ms per sim estimate. Hard-coded until a real backend reports it via
  // `window.BotEngine.benchmarkPerSim()`. Roughly: ~10ms on consumer GPU,
  // ~1-3ms on Blackwell, 50-200ms on CPU only.
  const msPerSim = (window.BotEngine && window.BotEngine.msPerSim) || 8.0;

  return (
    <div className="welcome">
      <div className="setup-card">
        <button className="back-btn" onClick={onBack}>← back</button>
        <div className="setup-title">new game</div>
        <div className="setup-section">
          <div className="setup-label">you play</div>
          <div className="setup-options">
            <button
              className={`opt opt-color ${color === 'w' ? 'opt-on' : ''}`}
              onClick={() => setColor('w')}
            >
              <span className="opt-swatch opt-swatch-w" />
              <span className="opt-name">white</span>
              <span className="opt-sub">you move first</span>
            </button>
            <button
              className={`opt opt-color ${color === 'b' ? 'opt-on' : ''}`}
              onClick={() => setColor('b')}
            >
              <span className="opt-swatch opt-swatch-b" />
              <span className="opt-name">black</span>
              <span className="opt-sub">bot moves first</span>
            </button>
            <button
              className={`opt opt-color ${color === '?' ? 'opt-on' : ''}`}
              onClick={() => setColor('?')}
            >
              <span className="opt-swatch opt-swatch-r"><span>?</span></span>
              <span className="opt-name">random</span>
              <span className="opt-sub">coin flip</span>
            </button>
          </div>
        </div>

        <div className="setup-section">
          <div className="setup-label">bot mode</div>
          <div className="setup-options setup-modes">
            <button
              className={`opt opt-mode ${mode === 'greedy' ? 'opt-on' : ''}`}
              onClick={() => setMode('greedy')}
            >
              <span className="opt-name">greedy</span>
              <span className="opt-sub">argmax · strongest</span>
              <span className="opt-tag">T=0</span>
            </button>
            <button
              className={`opt opt-mode ${mode === 't03' ? 'opt-on' : ''}`}
              onClick={() => setMode('t03')}
            >
              <span className="opt-name">sample</span>
              <span className="opt-sub">slightly varied</span>
              <span className="opt-tag">T=0.3</span>
            </button>
            <button
              className={`opt opt-mode ${mode === 't07' ? 'opt-on' : ''}`}
              onClick={() => setMode('t07')}
            >
              <span className="opt-name">loose</span>
              <span className="opt-sub">chill, makes mistakes</span>
              <span className="opt-tag">T=0.7</span>
            </button>
          </div>
        </div>

        <SimDepthPicker
          numSims={numSims}
          setNumSims={setNumSims}
          msPerSim={msPerSim}
        />

        <button
          className="play-btn play-btn-start"
          onClick={() => {
            const actualColor = color === '?' ? (Math.random() < 0.5 ? 'w' : 'b') : color;
            onStart({ color: actualColor, mode, numSims });
          }}
        >
          start game →
        </button>
      </div>
    </div>
  );
}

// ── Top toolbar ───────────────────────────────────────────────────────────
function TopBar({ playerColor, mode, onResign, onNewGame, onFlip, onAbout, gameOver }) {
  return (
    <div className="topbar">
      <div className="topbar-left">
        <a href="Hub.html" className="brand brand-small" style={{ textDecoration: 'none', color: 'inherit' }}>
          bottisme<span className="brand-dot">.</span>
        </a>
        <span className="crumb-sep" style={{ color: 'var(--muted)', margin: '0 4px' }}>/</span>
        <span className="topbar-meta">
          <span className="chip">you · {playerColor === 'w' ? 'white' : 'black'}</span>
          <span className="chip chip-mode">mode · {mode}</span>
        </span>
      </div>
      <div className="topbar-right">
        <a href="Hub.html" className="bar-btn" title="Back to all tools">↩ hub</a>
        <button className="bar-btn" onClick={onFlip} title="Flip board">⇅ flip</button>
        {!gameOver && <button className="bar-btn" onClick={onResign}>resign</button>}
        <button className="bar-btn bar-btn-primary" onClick={onNewGame}>+ new game</button>
      </div>
    </div>
  );
}

// ── Player chip (above/below board) ───────────────────────────────────────
function PlayerChip({ name, color, isBot, isTurn, captured }) {
  return (
    <div className={`player-chip ${isTurn ? 'turn' : ''}`}>
      <span className={`pc-dot pc-dot-${color}`} />
      <span className="pc-name">{name}</span>
      {isBot && <span className="pc-rating">~1850</span>}
      <span className="pc-captured">
        {captured.map((p, i) => (
          <Piece key={i} type={p.type} color={p.color} size={18} />
        ))}
      </span>
    </div>
  );
}

// ── Bot status (current thought) ──────────────────────────────────────────
function BotStatus({ thinking, lastBotMove, playerColor, evalFromYourSide }) {
  if (thinking) {
    return (
      <div className="bot-status thinking">
        <span className="thinking-dot" />
        <span className="thinking-dot" />
        <span className="thinking-dot" />
        <span className="bs-text">bot evaluating…</span>
      </div>
    );
  }
  if (lastBotMove) {
    const v = evalFromYourSide
      ? (playerColor === 'w' ? lastBotMove.vAfter : -lastBotMove.vAfter)
      : lastBotMove.vAfter;
    return (
      <div className="bot-status">
        <span className="bs-label">bot played</span>
        <span className="bs-san">{lastBotMove.move.san}</span>
        <span className="bs-pct">{Math.round(lastBotMove.prob * 100)}%</span>
        <span className={`bs-v ${v >= 0 ? 'pos' : 'neg'}`}>V = {formatV(v)}</span>
      </div>
    );
  }
  return (
    <div className="bot-status">
      <span className="bs-label">your move</span>
    </div>
  );
}

// ── Game over modal ───────────────────────────────────────────────────────
function GameOver({ result, reason, onNewGame, onClose, onDownload, onCopyPgn }) {
  const titles = {
    'win': 'you win',
    'loss': 'bot wins',
    'draw': 'draw',
  };
  return (
    <div className="modal-overlay">
      <div className="modal gameover">
        <div className="go-tag">{result === 'win' ? '1–0' : result === 'loss' ? '0–1' : '½–½'}</div>
        <div className="go-title">{titles[result]}</div>
        <div className="go-reason">{reason}</div>
        <div className="go-actions">
          <button className="bar-btn" onClick={onCopyPgn}>copy pgn</button>
          <button className="bar-btn" onClick={onDownload}>download pgn</button>
          <button className="bar-btn bar-btn-primary" onClick={onNewGame}>+ new game</button>
        </div>
        <button className="go-close" onClick={onClose}>✕</button>
      </div>
    </div>
  );
}

window.BottismeShell = { Welcome, Setup, TopBar, PlayerChip, BotStatus, GameOver };
