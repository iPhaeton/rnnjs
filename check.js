const {getSequences, forwardPass, backwardPass} = require('./rnn');

async function gradCheck(hiddenSize) {
  const [[sequence], sizes, weights] = await getSequences(hiddenSize);
  const [w, why, bh, by] = weights;
  const [h, probs, loss] = forwardPass(sequence, sizes, weights);
  const [_, dwa, dwhya, dbha, dbya] = backwardPass(probs, h, sequence, sizes, weights);

  console.log(dwa);
}

module.exports = {
  gradCheck,
}
