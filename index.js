const math = require('mathjs')
const {readText} = require('./utils');
const FILENAME = 'input.txt';
const EOS = '\n';

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

function stepForward(oneHotChar, oneHotY, [vocabSize, hiddenSize], [w, why, bh, by]) {
  hPrev = math.zeros(1, hiddenSize);
  const xh = math.concat(oneHotChar, hPrev);

  let h = math.multiply(w, math.reshape(xh, [123,1]));
  h = math.add(h, bh);
  h = math.tanh(h);
  let scores = math.multiply(why, h);
  scores = math.add(scores, by);

  return [h, scores];
}

async function rnn(hiddenSize) {
  const [inputs, vocabulary, charToIndex] = await readText(FILENAME, EOS);
  const sizes = [vocabulary.length, hiddenSize];
  const weights = initialize(sizes);
  const sequences = inputs.map(seq => oneHotEncode(seq, sizes[0], charToIndex));

  const [h, scores] = stepForward(sequences[0][0], sequences[0][1], sizes, weights);
  console.log(h);
  //sequences.forEach(seq => forwardPass(seq, sizes, weights))
}

rnn(100);
