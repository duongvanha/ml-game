const LogisticRegression = require('./LogisticRegression');
const loadData           = require('../regressionsv2/load-csv');

const {features, labels, testFeatures, testLabels} = loadData('regressionsv2/cars.csv', {
    shuffle     : true,
    splitTest   : 20,
    dataColumns : ['horsepower', 'weight', 'displacement'],
    labelColumns: ['passedemissions'],
    converters  : {
        passedemissions: (val) => val === 'TRUE' ? 1 : 0,
    },
});

let logisticRegression = new LogisticRegression(features, labels, {
    learningRate: .1,
    iterations  : 20,
    batchSize   : 10,
});

logisticRegression.train();
logisticRegression.predict(testFeatures).print();
console.log(testLabels);
