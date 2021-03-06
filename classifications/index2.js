const _                  = require('lodash');
const loadData           = require('../regressionsv2/load-csv');
const LogisticRegression = require('../logisticRegression/LogisticRegression');
const plot               = require('node-remote-plot');


const {features, labels, testFeatures, testLabels} = loadData('classifications/data1.csv', {
    shuffle     : true,
    splitTest   : 100,
    dataColumns : ['position', 'bounciness'],
    labelColumns: ['bucket\r'],
    converters  : {
        'bucket\r': (val) => {
            const arr              = new Array(10).fill(0);
            arr[parseInt(val) - 1] = 1;
            return arr
        },
    },
});


const logisticRegression = new LogisticRegression({
    learningRate: .1,
    iterations  : 20,
    batchSize   : 100,
});


logisticRegression.train();


plot({
    x     : logisticRegression.mseHistory.reverse(),
    xLabel: 'Iteration #',
    yLabel: 'Mean Squared Error',
});

