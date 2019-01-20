const math = require('mathjs');
const {getSequences, forwardPass, backwardPass} = require('./rnn');

function computeFunction(sequence, sizes, weights) {
  const [_, __, loss] = forwardPass(sequence, sizes, weights);
  const f = math.add(...loss, 0) / (sequence.length - 1);
  return f;
}

function computeGradient(sequence, sizes, weights, [k,i,j], h) {
  const weightsCopy = math.clone(weights);
  weightsCopy[k][i][j] += h;
  const fPlus = computeFunction(sequence, sizes, weightsCopy);
  weightsCopy[k][i][j] -= 2*h;
  const fMinus = computeFunction(sequence, sizes, weightsCopy);
  const grad = (fPlus - fMinus) / (2*h);
  return grad;
}

async function gradientCheck(hiddenSize) {
  const [[sequence], sizes, weights] = await getSequences(hiddenSize);
  const testSequence = sequence.slice(0,2);
  const [w, why, bh, by] = weights;
  const [h, probs, loss] = forwardPass(sequence, sizes, weights);
  const [_, dwa, dwhya, dbha, dbya] = backwardPass(probs, h, sequence, sizes, weights);

  console.log(dwa.toArray()[0][0], computeGradient(testSequence, sizes, weights, [0,0,0], 1e-5));

  // for (let k = 0; k < weights.length; k++) {
  //   const m = weights[k];
  //   for (let i = 0; i < m.length; i++) {
  //     const row = m[i];
  //     for (let j = 0; j < row.length; j++) {
  //       computeGradient(sequence, sizes, weights, [k,i,j]);
  //     }
  //   }
  // }
}

module.exports = {
  gradientCheck,
}
