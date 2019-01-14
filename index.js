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
    dotMultiply(math.reshape(math.clone(yOneHot), [scores.size()[0], 1])),
    math.log,
  )(probs);

  return loss;
}

function stepForward(charOneHot, yOneHot, [vocabSize, hiddenSize], [w, why, bh, by]) {
  hPrev = math.zeros(1, hiddenSize);
  const xh = math.concat(charOneHot, hPrev);

  const h = compose(
    math.tanh,
    add(bh),
    math.multiply,
  )(w, math.reshape(math.clone(xh), [123, 1]));

  const scores = compose(
    add(by),
    math.multiply,
  )(why, h);

  const loss = computeLoss(scores, yOneHot);

  return [h, loss];
}

function forwardPass(sequence, sizes, weights) {
  const [h, loss] = sequence
    .slice(0, -1)
    .reduce((res, charOneHot, i) => {
      const [h, loss] = stepForward(charOneHot, sequence[i+1], sizes, weights);
      return [[...res[0], h], res[1] + loss];
    }, [[], 0]);

  return [h, loss / (sequence.length - 1)];
}

async function rnn(hiddenSize) {
  const [inputs, vocabulary, charToIndex] = await readText(FILENAME, EOS);
  const sizes = [vocabulary.length, hiddenSize];
  const weights = initialize(sizes);
  const sequences = inputs.map(seq => oneHotEncode(seq, sizes[0], charToIndex));
  //const y = inputs.map(seq => seq.map(char => charToIndex[char]));

  const [h, loss] = forwardPass(sequences[0], sizes, weights);
  console.log(loss);
}

rnn(100);
