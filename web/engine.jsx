// engine.jsx — chess.js wrapper + mock bot inference
// The "bot" is a soft heuristic: material + tiny positional bonus + softmax-with-temp
// over a small set of features so top-K candidates look plausible without a real net.

const PIECE_VALUES = { p: 1, n: 3.0, b: 3.2, r: 5.0, q: 9.0, k: 0 };

// Piece-square heuristic. Different per piece type so the policy has visible
// preferences in the opening (centre pawns > rim pawns, knights to f3/c3, etc).
const CENTRE = [0, 0.2, 0.5, 0.9, 0.9, 0.5, 0.2, 0]; // by file or rank index

function squareBonus(piece, file, rank) {
  const centreScore = (CENTRE[file] + CENTRE[rank]) * 0.5; // 0..0.9
  const isOurStart = (piece.color === 'w' && rank === 0) || (piece.color === 'b' && rank === 7);

  switch (piece.type) {
    case 'p': {
      const adv = piece.color === 'w' ? rank : 7 - rank;  // 0..7
      // centre pawns reward advancement more than rim pawns
      const centreFile = CENTRE[file]; // 0..0.9
      return centreFile * (adv * 0.06) + (centreFile - 0.2) * 0.04;
    }
    case 'n': {
      // Knights love c3/f3 etc. Strong centre bonus, penalty on rim.
      return centreScore * 0.35 - (isOurStart ? 0.12 : 0);
    }
    case 'b': {
      // Bishops: bonus for being developed off back rank
      return centreScore * 0.22 - (isOurStart ? 0.08 : 0);
    }
    case 'r': {
      // Rooks: small bonus for centre files (open files)
      return CENTRE[file] * 0.08;
    }
    case 'q': {
      // Early queen development is bad. Reward staying home unless centred late.
      const backRank = piece.color === 'w' ? 0 : 7;
      if (rank === backRank && (file === 3)) return 0.05;
      // small penalty for being out early — proxy: penalise off-back-rank slightly
      return rank === backRank ? 0 : -0.06;
    }
    case 'k': {
      // King wants to be tucked into back rank (castling proxy).
      const backRank = piece.color === 'w' ? 0 : 7;
      if (rank !== backRank) return -0.35;
      // bonus for being on g/c (castled-ish positions)
      if (file === 6 || file === 2) return 0.15;
      if (file === 4) return 0; // starting square neutral
      return 0.04;
    }
  }
  return 0;
}

// V in [-1, 1] from white's perspective. Uses tanh of a scaled centipawn-ish score.
function evalPosition(chess) {
  if (chess.in_checkmate()) return chess.turn() === 'w' ? -1 : 1;
  if (chess.in_stalemate() || chess.insufficient_material() ||
      chess.in_threefold_repetition() || chess.in_draw()) return 0;

  let score = 0;
  const board = chess.board();
  for (let r = 0; r < 8; r++) {
    for (let f = 0; f < 8; f++) {
      const sq = board[r][f];
      if (!sq) continue;
      const sign = sq.color === 'w' ? 1 : -1;
      // r=0 is rank 8, r=7 is rank 1. Internal rank = 7 - r.
      const rank = 7 - r;
      score += sign * PIECE_VALUES[sq.type];
      score += sign * squareBonus(sq, f, rank);
    }
  }
  // Check is uncomfortable for the side to move.
  if (chess.in_check()) score += (chess.turn() === 'w' ? -0.25 : 0.25);

  return Math.tanh(score / 4.5);
}

function temperedSoftmax(scores, temperature) {
  if (temperature < 1e-4) {
    const max = Math.max(...scores);
    const maxIdx = scores.indexOf(max);
    return scores.map((_, i) => i === maxIdx ? 1 : 0);
  }
  const m = Math.max(...scores);
  const exps = scores.map(s => Math.exp((s - m) / temperature));
  const sum = exps.reduce((a, b) => a + b, 0);
  return exps.map(e => e / sum);
}

// Analyse the current position. Returns:
//   { vCurrent, scored: [{move, vAfter, prob, score}], topMoves }
// vAfter is from WHITE's perspective. score is from MOVER's perspective
// (what we softmax over).
function analyze(chess, temperature) {
  const vCurrent = evalPosition(chess);
  const moves = chess.moves({ verbose: true });
  if (!moves.length) return { vCurrent, scored: [], topMoves: [] };

  const tmp = new Chess();
  const scored = moves.map(m => {
    tmp.load(chess.fen());
    tmp.move(m);
    const vAfter = evalPosition(tmp);
    const moverSign = m.color === 'w' ? 1 : -1;
    return { move: m, vAfter, score: vAfter * moverSign };
  });

  // For the "policy" we use a fixed-ish softmax temperature so probabilities
  // look like a real policy head (not collapsed onto one move).
  const displayTemp = Math.max(temperature, 0.08);
  const probs = temperedSoftmax(scored.map(s => s.score), displayTemp);
  scored.forEach((s, i) => { s.prob = probs[i]; });
  scored.sort((a, b) => b.prob - a.prob);

  return {
    vCurrent,
    scored,
    topMoves: scored.slice(0, 5),
  };
}

// Sample a move from `scored` using sampling temperature.
function sampleMove(scored, temperature) {
  if (temperature < 1e-4) return scored[0]; // already sorted by display-prob
  // Re-softmax over `score` with the sampling temperature for fidelity.
  const probs = temperedSoftmax(scored.map(s => s.score), temperature);
  const r = Math.random();
  let cum = 0;
  for (let i = 0; i < scored.length; i++) {
    cum += probs[i];
    if (r <= cum) return scored[i];
  }
  return scored[scored.length - 1];
}

// Mode → sampling temperature
const MODE_TEMP = { greedy: 0, t03: 0.30, t07: 0.70 };

// Per-sim cost estimate. Used by the UI to display "~Xs thinking time" for a
// chosen MCTS sim count. Hard-coded for the prototype; the real backend should
// overwrite this on startup with a value measured by sampling_chess.mcts.benchmark_per_sim.
const msPerSim = 8.0;

window.BotEngine = { evalPosition, analyze, sampleMove, MODE_TEMP, msPerSim };
