const load             = require('./load-csv');
const LinearRegression = require('./linearRegression');

const {features, labels, testFeatures, testLabels} = load('./cars.csv', {
    shuffle     : true,
    splitTest   : 10,
    dataColumns : ['horsepower'],
    labelColumns: ['mpg'],
});


const linearRegression = new LinearRegression(features, labels, {
    learningRate: 0.001,
    iterations  : 100,
});


linearRegression.train();


console.log(`val m ${linearRegression.m}: , val b : ${linearRegression.b}`);
