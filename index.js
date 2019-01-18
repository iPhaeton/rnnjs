const math = require('mathjs')
const {readText, curry, compose} = require('./utils');
const FILENAME = 'input.txt';
const EOS = '\n';

const add = curry(math.add);
const subtract = curry(math.subtract);
const multiply = curry(math.multiply);
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
    dotMultiply(math.transpose(yOneHot)),
    math.log,
  )(probs);

  return [probs, loss];
}

function stepForward(charOneHot, hPrev, yOneHot, [vocabSize, hiddenSize], [w, why, bh, by]) {
  const xh = math.concat(charOneHot, math.transpose(hPrev));

  const h = compose(
    math.tanh,
    add(bh),
    math.multiply,
  )(w, math.transpose(xh));

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

function stepBackward(probs, x, y, hs, sizes, weights) {
  const [w, why, bh, by] = weights;
  const [vocabSize, hiddenSize] = sizes;
  const [hPrev, h, dhNext] = hs;

  const dy = math.subtract(probs, math.transpose(y));

  const dwhy = math.multiply(dy, math.transpose(h));
  const dby = math.clone(dy);

  const dh = compose(
    add(dhNext),
    math.multiply,
  )(math.transpose(why), dy);

  const dhRaw = compose(
    dotMultiply(dh),
    subtract(1),
    math.square,
  )(h);

  const dw = math.multiply(dhRaw, math.concat(x, math.transpose(hPrev)));
  const dbh = math.clone(dhRaw);

  const dhPrev = math.multiply(math.transpose(w).slice(vocabSize, vocabSize + hiddenSize), dhRaw);
  return [dhPrev, dw, dbh, dwhy, dby];
}

function backwardPass(probs, h, sequence, sizes, weights) {
  const [vocabSize, hiddenSize] = sizes;

  probs.reduceRight((res, prob, i) => {
    const [dhPrev, dw, dbh, dwhy, dby] = stepBackward(
      probs[i],
      sequence[i],
      sequence[i+1],
      [
        h[i],
        h[i+1],
        res[0],
      ],
      sizes,
      weights,
    );
    return res;
  }, [math.zeros(hiddenSize, 1)]);
}

async function rnn(hiddenSize) {
  const [inputs, vocabulary, charToIndex] = await readText(FILENAME, EOS);
  const sizes = [vocabulary.length, hiddenSize];
  const weights = initialize(sizes);
  const sequences = inputs.map(seq => oneHotEncode(seq, sizes[0], charToIndex));

  const [h, probs, loss] = forwardPass(sequences[0], sizes, weights);
  backwardPass(probs, h, sequences[0], sizes, weights)

  // const [dhPrev, dw, dbh, dwhy, dby] = stepBackward(
  //   probs[12],
  //   sequences[0][12],
  //   sequences[0][13],
  //   [
  //     h[12],
  //     h[13],
  //     math.zeros(hiddenSize, 1)
  //   ],
  //   sizes,
  //   weights,
  // );
}

rnn(100);
