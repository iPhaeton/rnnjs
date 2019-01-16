const math = require('mathjs')
const {readText, curry, compose} = require('./utils');
const FILENAME = 'input.txt';
const EOS = '\n';

const add = curry(math.add);
const dotMultiply = curry(math.dotMultiply);

function initialize([vocabSize, hiddenSize]) {
  const w = math.multiply(math.random([hiddenSize, vocabSize + hiddenSize]), 0.01);
  const why = math.multiply(math.random([vocabSize, hiddenSize]), 0.01);
  const bh = math.zeros(hiddenSize, 1);
  const by = math.zeros(vocabSize, 1);

  return [w, why, bh, by];
}

function oneHotEncode(sequence, vocabSize, charToIndex) {
  return sequence.map(char => {
    const encoding = math.zeros(1, vocabSize);
    encoding.subset(math.index(0, charToIndex[char]), 1);
    return encoding;
  })
}

function computeLoss(scores, yOneHot) {
  const exp = math.exp(scores);
  const sum = math.sum(exp);
  const probs = math.divide(exp, sum);

  const loss = -compose(
    math.sum,
    dotMultiply(math.clone(yOneHot).reshape([scores.size()[0], 1])),
    math.log,
  )(probs);

  return [probs, loss];
}

function stepForward(charOneHot, hPrev, yOneHot, [vocabSize, hiddenSize], [w, why, bh, by]) {
  const xh = math.concat(charOneHot, math.clone(hPrev).reshape([1, hiddenSize]));

  const h = compose(
    math.tanh,
    add(bh),
    math.multiply,
  )(w, math.clone(xh).reshape([123, 1]));

  const scores = compose(
    add(by),
    math.multiply,
  )(why, h);

  const [probs, loss] = computeLoss(scores, yOneHot);

  return [h, probs, loss];
}

function forwardPass(sequence, sizes, weights) {
  const [vocabSize, hiddenSize] = sizes;
  const [h, probs, loss] = sequence
    .slice(0, -1)
    .reduce((res, charOneHot, i) => {
      const [h, probs, loss] = stepForward(charOneHot, res[0][i], sequence[i + 1], sizes, weights);
      return [
        [...res[0], h],
        [...res[1], probs],
        [...res[2], loss],
      ];
    }, [[math.zeros(hiddenSize, 1)], [], []]);

  return [h, probs, loss];
}

function stepBackward(loss, probs, y) {
  const dy = compose(
    add(-1),
    math.sum,
    dotMultiply,
  )(math.clone(probs).reshape([1, probs.size()[0]]), y);

  //const dwhy =
  //const dh =

  //const dw =
  //const dhPrev =

  console.log(dy);
}

async function rnn(hiddenSize) {
  const [inputs, vocabulary, charToIndex] = await readText(FILENAME, EOS);
  const sizes = [vocabulary.length, hiddenSize];
  const weights = initialize(sizes);
  const sequences = inputs.map(seq => oneHotEncode(seq, sizes[0], charToIndex));

  const [h, probs, loss] = forwardPass(sequences[0], sizes, weights);

  stepBackward(loss[12], probs[12], sequences[0][13])
}

rnn(100);
