const load             = require('./load-csv');
const LinearRegression = require('./linearRegression');
const plot             = require('node-remote-plot');

const {features, labels, testFeatures, testLabels} = load('./cars.csv', {
    shuffle     : true,
    splitTest   : 50,
    dataColumns : ['horsepower', 'weight', 'displacement'],
    labelColumns: ['mpg'],
});


const linearRegression = new LinearRegression(features, labels, {
    learningRate: .1,
    iterations  : 20,
    batchSize   : 10,
});


linearRegression.train();


console.log(linearRegression.test(testFeatures, testLabels));

plot({
    x     : linearRegression.mseHistory.reverse(),
    xLabel: 'Iteration #',
    yLabel: 'Mean Squared Error',
});
