const {readText} = require('./utils');
const FILENAME = 'input.txt';
const EOS = '\n';

async function rnn() {
  const [inputs, vocabulary, charToIndex] = await readText(FILENAME, EOS)
}

rnn()