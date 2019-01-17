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

const curry = (func) => {
  return function(...args) {
    let storedArgs = [];

    function f (...args) {
      args.forEach(a => storedArgs.push(a))
      if (storedArgs.length < func.length) {
        return f;
      } else {
        const res = func(...storedArgs);
        //storedArgs = [];
        return res;
      }
    }

    return f(...args);
  }
};

const compose = (...funcs) => (...initialArgs) => {
  return funcs.reduceRight((prevResult, f) => [f(...prevResult)], initialArgs)[0];
}

module.exports = {
  promisify,
  readFile,
  readText,
  compose,
  curry,
};
