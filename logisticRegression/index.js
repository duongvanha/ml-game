const LogisticRegression = require('./LogisticRegression');
const _                  = require('lodash');
const loadData           = require('../regressionsv2/load-csv');

const {features, labels, testFeatures, testLabels} = loadData('regressionsv2/cars.csv', {
    shuffle     : true,
    splitTest   : 20,
    dataColumns : ['horsepower', 'weight', 'displacement'],
    labelColumns: ['mpg'],
    converters  : {
        mpg: (val) => {
            if (val <= 15) return [1, 0, 0];
            if (val <= 30) return [0, 1, 0];
            return [0, 0, 1];
        },
    },
});

let logisticRegression = new LogisticRegression(features, _.flatMap(labels), {
    learningRate: .1,
    iterations  : 20,
    batchSize   : 10,
});

logisticRegression.train();

console.log(logisticRegression.test(testFeatures, _.flatMap(testLabels)));
