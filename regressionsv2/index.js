const load             = require('./load-csv');
const LinearRegression = require('./linearRegression');

const {features, labels, testFeatures, testLabels} = load('./cars.csv', {
    shuffle     : true,
    splitTest   : 50,
    dataColumns : ['horsepower'],
    labelColumns: ['mpg'],
});


const linearRegression = new LinearRegression(features, labels, {
    learningRate: 0.1,
    iterations  : 100,
});


linearRegression.train();


console.log(linearRegression.test(testFeatures, testLabels));
