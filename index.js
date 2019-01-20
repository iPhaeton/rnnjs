const {rnn} = require('./rnn');
const {gradientCheck} = require('./check');

async function checkGradients() {
  const errors = await gradientCheck(10);
  console.log(`
    dw error: ${errors[0]}; 
    dwhy error ${errors[1]}; 
    dbh error ${errors[2]}; 
    dby error ${errors[3]};`
  );
}

rnn(100)
checkGradients();