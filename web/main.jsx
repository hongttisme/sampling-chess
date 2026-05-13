// main.jsx — Bottisme app root: state machine, bot orchestration, glue.

const { Welcome, Setup, TopBar, PlayerChip, BotStatus, GameOver } = window.BottismeShell;

const { Board, EvalBar, TopMovesPanel, MoveListPanel, PromotionModal, VTrendPanel } = window.BottismeUI;
const TWEAK_DEFAULTS_LOCAL = window.useBottismeTweakDefaults();

// PGN helpers
function buildPgn(history, color, mode, result) {
  const date = new Date().toISOString().slice(0,10).replace(/-/g, '.');
  const headers = [
    `[Event "Bottisme casual"]`,
    `[Site "bottisme.app"]`,
    `[Date "${date}"]`,
    `[Round "1"]`,
    `[White "${color === 'w' ? 'you' : 'bottisme'}"]`,
    `[Black "${color === 'w' ? 'bottisme' : 'you'}"]`,
    `[Result "${result === 'win' ? (color === 'w' ? '1-0' : '0-1') : result === 'loss' ? (color === 'w' ? '0-1' : '1-0') : result === 'draw' ? '1/2-1/2' : '*'}"]`,
    `[BotMode "${mode}"]`,
  ].join('\n');
  let body = '';
  for (let i = 0; i < history.length; i += 2) {
    body += `${(i/2) + 1}. ${history[i].san}`;
    if (history[i+1]) body += ` ${history[i+1].san}`;
    body += ' ';
  }
  return headers + '\n\n' + body.trim();
}

// Find captures for "captured pieces" strip
function capturedSets(history) {
  // history: ply objects, each with .move (verbose chess.js move)
  const capturedByWhite = []; // pieces captured BY white (i.e. black pieces)
  const capturedByBlack = [];
  history.forEach(h => {
    const m = h.move;
    if (!m) return;
    if (m.captured) {
      const cap = { type: m.captured, color: m.color === 'w' ? 'b' : 'w' };
      if (m.color === 'w') capturedByWhite.push(cap);
      else capturedByBlack.push(cap);
    }
  });
  return { capturedByWhite, capturedByBlack };
}

// Detect game over reason
function gameOverInfo(chess, playerColor) {
  if (chess.in_checkmate()) {
    const loser = chess.turn();
    const result = loser === playerColor ? 'loss' : 'win';
    return { result, reason: `checkmate · ${loser === 'w' ? 'black' : 'white'} delivers mate` };
  }
  if (chess.in_stalemate()) return { result: 'draw', reason: 'stalemate' };
  if (chess.insufficient_material()) return { result: 'draw', reason: 'insufficient material' };
  if (chess.in_threefold_repetition()) return { result: 'draw', reason: 'threefold repetition' };
  if (chess.in_draw()) return { result: 'draw', reason: 'fifty-move rule' };
  return null;
}

// ── Root App ──────────────────────────────────────────────────────────────
function App() {
  const [t, setTweak] = useTweaks(TWEAK_DEFAULTS_LOCAL);
  const [screen, setScreen] = React.useState('welcome'); // 'welcome' | 'setup' | 'game'

  // Game config
  const [playerColor, setPlayerColor] = React.useState('w');
  const [mode, setMode] = React.useState('greedy');
  const [numSims, setNumSims] = React.useState(200);

  // Live game state
  const chessRef = React.useRef(null);
  const [fen, setFen] = React.useState(new Chess().fen());
  const [history, setHistory] = React.useState([]); // [{move, san, color, vAfter, prob, mistake}]
  const [analysis, setAnalysis] = React.useState(null);
  const [thinking, setThinking] = React.useState(false);
  const [lastBotMove, setLastBotMove] = React.useState(null);
  const [orientation, setOrientation] = React.useState('w');

  // Interaction state
  const [selected, setSelected] = React.useState(null);
  const [legalTargets, setLegalTargets] = React.useState(new Set());
  const [pendingPromotion, setPendingPromotion] = React.useState(null); // {from, to, color}
  const [hoveredHint, setHoveredHint] = React.useState(null);

  // End state
  const [overState, setOverState] = React.useState(null); // { result, reason }
  const [overDismissed, setOverDismissed] = React.useState(false);

  // ── Game lifecycle ───────────────────────────────────────────────────────
  const startGame = React.useCallback(({ color, mode: m, numSims: ns }) => {
    chessRef.current = new Chess();
    setPlayerColor(color);
    setMode(m);
    if (ns !== undefined) setNumSims(ns);
    setOrientation(color);
    setFen(chessRef.current.fen());
    setHistory([]);
    setAnalysis(window.BotEngine.analyze(chessRef.current, window.BotEngine.MODE_TEMP[m]));
    setLastBotMove(null);
    setSelected(null);
    setLegalTargets(new Set());
    setOverState(null);
    setOverDismissed(false);
    setScreen('game');
  }, []);

  // Re-analyze position whenever fen changes
  React.useEffect(() => {
    if (!chessRef.current || screen !== 'game') return;
    const a = window.BotEngine.analyze(chessRef.current, window.BotEngine.MODE_TEMP[mode]);
    setAnalysis(a);
  }, [fen, mode, screen]);

  // Bot move scheduling: if it's bot's turn and no game over, play after a delay
  React.useEffect(() => {
    if (screen !== 'game' || !chessRef.current) return;
    const turn = chessRef.current.turn();
    if (turn === playerColor) return;
    if (overState) return;

    setThinking(true);
    const timer = setTimeout(() => {
      const a = window.BotEngine.analyze(chessRef.current, window.BotEngine.MODE_TEMP[mode]);
      if (a.scored.length === 0) {
        setThinking(false);
        return;
      }
      const pick = window.BotEngine.sampleMove(a.scored, window.BotEngine.MODE_TEMP[mode]);
      // Apply move
      chessRef.current.move({
        from: pick.move.from, to: pick.move.to,
        promotion: pick.move.promotion || undefined,
      });
      const newFen = chessRef.current.fen();
      const ply = {
        move: pick.move,
        san: pick.move.san,
        color: pick.move.color,
        vAfter: pick.vAfter,
        prob: pick.prob,
        mistake: false,
      };
      setHistory(h => [...h, ply]);
      setFen(newFen);
      setLastBotMove(ply);
      setThinking(false);
      const over = gameOverInfo(chessRef.current, playerColor);
      if (over) setOverState(over);
      // Mock thinking-time scales with sim count (real backend will block on
      // the actual MCTS call instead).
    }, Math.max(300, numSims * ((window.BotEngine && window.BotEngine.msPerSim) || 8.0)) + Math.random() * 200);

    return () => { clearTimeout(timer); setThinking(false); };
  }, [fen, screen, playerColor, mode, numSims, overState]);

  // ── User interaction ─────────────────────────────────────────────────────
  const handleSquareClick = React.useCallback((sq) => {
    if (!chessRef.current || overState) return;
    if (chessRef.current.turn() !== playerColor) return;
    const piece = chessRef.current.get(sq);

    if (selected) {
      // try to move
      if (sq === selected) { setSelected(null); setLegalTargets(new Set()); return; }
      if (legalTargets.has(sq)) {
        // detect promotion
        const movingPiece = chessRef.current.get(selected);
        const needsPromo = movingPiece && movingPiece.type === 'p' &&
          ((movingPiece.color === 'w' && sq[1] === '8') || (movingPiece.color === 'b' && sq[1] === '1'));
        if (needsPromo) {
          setPendingPromotion({ from: selected, to: sq, color: movingPiece.color });
          return;
        }
        commitMove(selected, sq, null);
        return;
      }
      // switching to another own piece
      if (piece && piece.color === playerColor) {
        selectSquare(sq);
        return;
      }
      setSelected(null);
      setLegalTargets(new Set());
      return;
    }

    if (piece && piece.color === playerColor) selectSquare(sq);
  }, [selected, legalTargets, playerColor, overState]);

  const selectSquare = (sq) => {
    const moves = chessRef.current.moves({ square: sq, verbose: true });
    setSelected(sq);
    setLegalTargets(new Set(moves.map(m => m.to)));
  };

  const commitMove = (from, to, promotion) => {
    const moveObj = { from, to };
    if (promotion) moveObj.promotion = promotion;
    const result = chessRef.current.move(moveObj);
    if (!result) {
      setSelected(null); setLegalTargets(new Set());
      return;
    }
    const newFen = chessRef.current.fen();
    // Compute "mistake" — compare to bot's top1 for white's-perspective
    let mistake = false;
    if (analysis && analysis.topMoves.length) {
      const top = analysis.topMoves[0];
      // Both vAfter are from white's perspective. Mover-perspective diff:
      const moverSign = result.color === 'w' ? 1 : -1;
      // Find the V_after for the move the user just made
      const userScored = analysis.scored.find(s =>
        s.move.from === from && s.move.to === to && s.move.promotion === result.promotion);
      const userV = userScored ? userScored.vAfter : 0;
      const diff = (top.vAfter - userV) * moverSign;
      if (diff > 0.25) mistake = true;
    }
    const ply = {
      move: result,
      san: result.san,
      color: result.color,
      vAfter: window.BotEngine.evalPosition(chessRef.current),
      prob: 0,
      mistake,
    };
    setHistory(h => [...h, ply]);
    setFen(newFen);
    setSelected(null);
    setLegalTargets(new Set());
    setLastBotMove(null);
    const over = gameOverInfo(chessRef.current, playerColor);
    if (over) setOverState(over);
  };

  const handlePromoPick = (type) => {
    if (!pendingPromotion) return;
    const { from, to } = pendingPromotion;
    setPendingPromotion(null);
    commitMove(from, to, type);
  };

  // ── Toolbar handlers ─────────────────────────────────────────────────────
  const handleNewGame = () => { setScreen('setup'); setOverState(null); setOverDismissed(false); };
  const handleResign = () => {
    if (!chessRef.current || overState) return;
    setOverState({ result: 'loss', reason: 'you resigned' });
  };
  const handleFlip = () => { setOrientation(o => o === 'w' ? 'b' : 'w'); };

  const getPgn = () => buildPgn(history, playerColor, mode, overState ? overState.result : null);
  const handleDownloadPgn = () => {
    const pgn = getPgn();
    const blob = new Blob([pgn], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `bottisme-game-${Date.now()}.pgn`;
    a.click();
    URL.revokeObjectURL(url);
  };
  const handleCopyPgn = () => { navigator.clipboard.writeText(getPgn()); };

  // ── Render ───────────────────────────────────────────────────────────────
  // Apply theme classes
  React.useEffect(() => {
    document.documentElement.dataset.theme = t.theme;
    document.documentElement.dataset.density = t.density;
  }, [t.theme, t.density]);

  const isYourTurn = chessRef.current && chessRef.current.turn() === playerColor && !overState;

  // Hint squares: highlight bot's #1 recommendation when it's your turn
  let hintSquares = new Set();
  if (t.showHints && isYourTurn && analysis && analysis.topMoves.length) {
    const top = analysis.topMoves[0];
    hintSquares.add(top.move.from);
    hintSquares.add(top.move.to);
  }
  // When hovering a top move, override hint with that move
  if (hoveredHint) {
    hintSquares = new Set([hoveredHint.from, hoveredHint.to]);
  }

  const inCheckSquare = (() => {
    if (!chessRef.current || !chessRef.current.in_check()) return null;
    const turn = chessRef.current.turn();
    const board = chessRef.current.board();
    for (let r = 0; r < 8; r++) for (let f = 0; f < 8; f++) {
      const p = board[r][f];
      if (p && p.color === turn && p.type === 'k') return sqName(f, 7 - r);
    }
    return null;
  })();

  const { capturedByWhite, capturedByBlack } = capturedSets(history);

  // Piece size depends on density
  const pieceSize = t.density === 'compact' ? 46 : 56;

  if (screen === 'welcome') {
    return (
      <div className="root">
        <Welcome onStart={() => setScreen('setup')} />
      </div>
    );
  }
  if (screen === 'setup') {
    return (
      <div className="root">
        <Setup onStart={startGame} onBack={() => setScreen('welcome')} />
      </div>
    );
  }

  // Game screen
  const lastMove = history.length ? history[history.length - 1].move : null;
  const topPlayer = orientation === 'w' ? 'b' : 'w'; // who is shown on top
  const youOnTop = topPlayer === playerColor;

  return (
    <div className="root">
      <TopBar
        playerColor={playerColor}
        mode={mode}
        onResign={handleResign}
        onNewGame={handleNewGame}
        onFlip={handleFlip}
        gameOver={!!overState}
      />
      <div className={`game-grid ${!t.showEvalBar ? 'no-eval' : ''}`}>
        <div className="game-left">
          {t.showEvalBar && (
            <EvalBar
              v={analysis ? analysis.vCurrent : 0}
              playerColor={playerColor}
              evalFromYourSide={t.evalFromYourSide}
            />
          )}
        </div>

        <div className="game-center">
          <PlayerChip
            name={topPlayer === playerColor ? 'you' : 'bottisme'}
            color={topPlayer}
            isBot={topPlayer !== playerColor}
            isTurn={chessRef.current && chessRef.current.turn() === topPlayer && !overState}
            captured={topPlayer === 'w' ? capturedByWhite : capturedByBlack}
          />
          <Board
            fen={fen}
            orientation={orientation}
            selected={selected}
            legalTargets={legalTargets}
            lastMove={lastMove}
            onSquareClick={handleSquareClick}
            inCheckSquare={inCheckSquare}
            hintSquares={hintSquares}
            boardPalette={t.boardPalette}
            coordinates={t.coordinates}
            highlightLastMove={t.highlightLastMove}
            pieceSize={pieceSize}
          />
          <PlayerChip
            name={orientation === playerColor ? 'you' : 'bottisme'}
            color={orientation}
            isBot={orientation !== playerColor}
            isTurn={chessRef.current && chessRef.current.turn() === orientation && !overState}
            captured={orientation === 'w' ? capturedByWhite : capturedByBlack}
          />
        </div>

        <div className="game-right">
          <BotStatus
            thinking={thinking}
            lastBotMove={lastBotMove}
            playerColor={playerColor}
            evalFromYourSide={t.evalFromYourSide}
          />
          <TopMovesPanel
            analysis={analysis}
            isYourTurn={isYourTurn}
            thinking={thinking}
            playerColor={playerColor}
            evalFromYourSide={t.evalFromYourSide}
            onHover={(m) => setHoveredHint({ from: m.from, to: m.to })}
            onLeave={() => setHoveredHint(null)}
          />
          <MoveListPanel
            history={history}
            playerColor={playerColor}
            evalFromYourSide={t.evalFromYourSide}
          />
          <VTrendPanel
            history={history}
            playerColor={playerColor}
            evalFromYourSide={t.evalFromYourSide}
          />
        </div>
      </div>

      {pendingPromotion && (
        <PromotionModal
          color={pendingPromotion.color}
          onPick={handlePromoPick}
          onCancel={() => setPendingPromotion(null)}
        />
      )}

      {overState && !overDismissed && (
        <GameOver
          result={overState.result}
          reason={overState.reason}
          onNewGame={handleNewGame}
          onClose={() => setOverDismissed(true)}
          onDownload={handleDownloadPgn}
          onCopyPgn={handleCopyPgn}
        />
      )}

      <TweaksPanel title="Tweaks">
        <TweakSection label="theme">
          <TweakRadio
            label="surface"
            value={t.theme}
            options={[
              { label: 'paper', value: 'paper' },
              { label: 'mono', value: 'mono' },
              { label: 'ink', value: 'ink' },
            ]}
            onChange={(v) => setTweak('theme', v)}
          />
          <TweakSelect
            label="board palette"
            value={t.boardPalette}
            options={[
              { label: 'classic (wood)', value: 'classic' },
              { label: 'mono ink', value: 'monoink' },
              { label: 'green felt', value: 'felt' },
              { label: 'muted blue', value: 'blue' },
            ]}
            onChange={(v) => setTweak('boardPalette', v)}
          />
        </TweakSection>
        <TweakSection label="layout">
          <TweakRadio
            label="density"
            value={t.density}
            options={[
              { label: 'cozy', value: 'cozy' },
              { label: 'compact', value: 'compact' },
            ]}
            onChange={(v) => setTweak('density', v)}
          />
          <TweakToggle
            label="coordinate labels"
            value={t.coordinates}
            onChange={(v) => setTweak('coordinates', v)}
          />
          <TweakToggle
            label="highlight last move"
            value={t.highlightLastMove}
            onChange={(v) => setTweak('highlightLastMove', v)}
          />
        </TweakSection>
        <TweakSection label="info">
          <TweakToggle
            label="show eval bar"
            value={t.showEvalBar}
            onChange={(v) => setTweak('showEvalBar', v)}
          />
          <TweakToggle
            label="show hint (bot's pick)"
            value={t.showHints}
            onChange={(v) => setTweak('showHints', v)}
          />
          <TweakToggle
            label="V values from your side"
            value={t.evalFromYourSide}
            onChange={(v) => setTweak('evalFromYourSide', v)}
          />
        </TweakSection>
      </TweaksPanel>
    </div>
  );
}

// helpers exposed up here for game.jsx components
const FILES_M = ['a','b','c','d','e','f','g','h'];
function sqName(file, rank) { return FILES_M[file] + (rank + 1); }
function formatV(v) {
  if (v === null || v === undefined || isNaN(v)) return '–';
  const sign = v >= 0 ? '+' : '−';
  return sign + Math.abs(v).toFixed(2);
}

ReactDOM.createRoot(document.getElementById('root')).render(<App />);
