const math = require('mathjs')
const {readText} = require('./utils');
const FILENAME = 'input.txt';
const EOS = '\n';

function initialize(vocabSize, hiddenSize) {
  const w = math.multiply(math.ones(hiddenSize, vocabSize + hiddenSize), math.random() * 0.01);
  const why = math.multiply(math.ones(vocabSize, hiddenSize), math.random() * 0.01);

  return [w, why];
}

function oneHotEncode(sequence, vocabSize, charToIndex) {
  return sequence.map(char => {
    const encoding = math.zeros(1, vocabSize);
    encoding.subset(math.index(0, charToIndex[char]), 1);
    return encoding;
  })
}

async function rnn(hiddenSize) {
  const [inputs, vocabulary, charToIndex] = await readText(FILENAME, EOS);
  const [w, why] = initialize(vocabulary.length, hiddenSize);
  const sequences = inputs.map(seq => oneHotEncode(seq, vocabulary.length, charToIndex));

  console.log(sequences[0][1])
}

rnn(100);
