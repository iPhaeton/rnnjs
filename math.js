const {curry} = require('./utils');
const math = require('mathjs');

const add = curry(math.add);
const subtract = curry(math.subtract);
const multiply = curry(math.multiply);
const dotMultiply = curry(math.dotMultiply);

module.exports = {
  add,
  subtract,
  multiply,
  dotMultiply
};
