const fs = require('fs');
const path = require('path');

function promisify(f, context = null, errorFirst = true) {
  const ctx = context || this;
  return function (...args) {
    return new Promise((resolve, reject) => {
      f.call(ctx, ...arguments, (...args) => {
        const err = arguments ? args.find((a) => a instanceof Error) : null;
        if (err) {
          reject(err);
        } else if (errorFirst) {
          resolve(args.slice(1));
        } else {
          resolve(args)
        }
      })
    });
  }
};

const readFile = promisify(fs.readFile, fs);

async function readText(filename, eos) {
  const [text] = await readFile(path.resolve(__dirname, filename), 'utf8');

  const inputs = text
    .split(eos)
    .filter(seq => !!seq.length)
    .map(seq => `${seq}${eos}`)
    .map(seq => seq.split(''));

  const vocabulary = [...inputs.reduce((set, seq) => {
    seq.forEach(ch => set.add(ch));
    return set;
  }, new Set())];

  const charToIndex = vocabulary.reduce((res, ch, index) => ({...res, [ch]: index}), {});

  return [inputs, vocabulary, charToIndex];
};

module.exports = {
  promisify,
  readFile,
  readText,
};
