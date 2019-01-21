const math = require('mathjs');
const {readText, compose, mapZip} = require('./utils');
const FILENAME = 'input.txt';
const EOS = '\n';
const {add, subtract, multiply, dotMultiply} = require('./math');

const applyOnEveryStep = (f) => (...initialArgs) => (...args) => f(...initialArgs, ...args);

function initialize([vocabSize, hiddenSize], mode = ['random', 'zeros']) {
  if (!mode || mode.length !== 2) {
    throw new Error('[initialize] There should be mode for weights and for biases');
  }

  const [modeW, modeB] = mode;

  const w = compose(
    multiply(0.01),
    math[modeW],
  )([hiddenSize, vocabSize + hiddenSize]);

  const why = compose(
    multiply(0.01),
    math[modeW],
  )([vocabSize, hiddenSize]);

  const bh = math[modeB]([hiddenSize, 1]);
  const by = math[modeB]([vocabSize, 1]);

  return [w, why, bh, by];
}

function update(weights, gradients, lr) {
  return mapZip(compose(
    v => v.toArray(),
    applyOnEveryStep(multiply)(lr),
    subtract,
  ), weights, gradients);
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

function stepForward(charOneHot, hPrev, yOneHot, [w, why, bh, by]) {
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
      const [h, probs, loss] = stepForward(charOneHot, res[0][i], sequence[i + 1], weights);
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

  return probs.reduceRight((nextStep, prob, i) => {
    const [dhNext, dwNext, dwhyNext, dbhNext, dbyNext] = nextStep;

    const [dhPrev, dw, dbh, dwhy, dby] = stepBackward(
      probs[i],
      sequence[i],
      sequence[i+1],
      [
        h[i],
        h[i+1],
        dhNext,
      ],
      sizes,
      weights,
    );

    return [
      dhPrev,
      add(dwNext, dw),
      add(dwhyNext, dwhy),
      add(dbhNext, dbh),
      add(dbyNext, dby),
    ];
  }, [math.zeros(hiddenSize, 1), ...initialize(sizes, ['zeros', 'zeros'])]);
}

function log({probsHistory, lossHistory, sizes, weights, lr}) {
  console.log(lossHistory[lossHistory.length - 1])
}

function forwardAndBackward(config, sequence) {
  const {probsHistory, lossHistory, sizes, weights, lr} = config;

  const [h, probs, loss] = forwardPass(sequence, sizes, weights);
  const gradients = backwardPass(probs, h, sequence, sizes, weights);
  const meanLoss = math.add(...loss) / (sequence.length - 1);

  const updatedConfig = {
    ...config,
    probsHistory: [...probsHistory, probs],
    lossHistory: [...lossHistory, meanLoss],
    weights: update(weights, gradients.slice(1), lr),
  };

  log(updatedConfig);

  return updatedConfig;
}

async function getSequences(hiddenSize, filename) {
  const [inputs, vocabulary, charToIndex] = await readText(filename, EOS);
  const sizes = [vocabulary.length, hiddenSize];
  const weights = initialize(sizes, ['random', 'zeros']);
  const sequences = inputs.map(seq => oneHotEncode(seq, sizes[0], charToIndex));
  return [sequences, sizes, weights];
}

async function rnn(hiddenSize, lr = 0.01) {
  const [sequences, sizes, weights] = await getSequences(hiddenSize, FILENAME);

  const initialConfig = {
    probsHistory: [],
    lossHistory: [],
    sizes,
    weights,
    lr
  };

  const {lossHistory} = sequences.reduce(forwardAndBackward, initialConfig);
}

module.exports = {
  rnn,
  getSequences,
  forwardPass,
  backwardPass,
  initialize,
}
