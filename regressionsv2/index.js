const load             = require('./load-csv');
const LinearRegression = require('./linearRegression2');

const {features, labels, testFeatures, testLabels} = load('regressionsv2/cars.csv', {
    shuffle     : true,
    splitTest   : 10,
    dataColumns : ['horsepower', 'weight', 'displacement'],
    labelColumns: ['mpg'],
});


const linearRegression = new LinearRegression({
    learningRate: .02,
    iterations  : 20,
    batchSize   : 10,
});


linearRegression.train(features, labels).then(() => {
    linearRegression.predict(testFeatures).print();

    linearRegression.model.save('file://regressionsv2/model').then(() => console.log('saved'));

    console.log(testLabels);
});

