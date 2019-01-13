const math = require('mathjs')
const {readText} = require('./utils');
const FILENAME = 'input.txt';
const EOS = '\n';

function initialize(vocabSize, hiddenSize) {
  const wxh = math.multiply(math.ones(hiddenSize, vocabSize), math.random() * 0.01);
  const whh = math.multiply(math.ones(hiddenSize, hiddenSize), math.random() * 0.01);
  const why = math.multiply(math.ones(vocabSize, hiddenSize), math.random() * 0.01);

  return [wxh, whh, why];
}

async function rnn(hiddenSize) {
  const [inputs, vocabulary, charToIndex] = await readText(FILENAME, EOS);
  const [wxh, whh, why] = initialize(vocabulary.length, hiddenSize);
}

rnn(100);
