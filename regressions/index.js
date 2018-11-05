require('@tensorflow/tfjs-node');
const load = require('./load-csv');
const tf = require('@tensorflow/tfjs');



const { features, labels, testFeatures, testLabels } = load('./data.csv', {
    shuffle     : true,
    splitTest   : 10,
    dataColumns : ['lat', 'long'],
    labelColumns: ['price']
})

const _features = tf.tensor(features);
const _labels = tf.tensor(labels);


function knn(features, labels, predictionPoint, k) {
    return features
        .sub(predictionPoint)
        .pow(2)
        .sum(1)
        .pow(0.5)
        .expandDims(1)
        .concat(labels, 1)
        .unstack()
        .sort((a, b) => a.get(0) - b.get(0))
        .slice(0, k)
        .map(i => i.get(1))
        .reduce((total, i) => total + i, 0) / k
}

console.log(knn(_features, _labels, tf.tensor(testFeatures[0]), 2),' - ',testLabels[0][0])