const parser        = require('csv-parse/lib/sync');
const fs            = require('fs');
const _             = require('lodash');
const trainer       = require('./train');

const rawData       = fs.readFileSync('data.csv').toString();

function splitDataSet(data, testCount) {
    const shuffled  = _.shuffle(data);
    const dataTest  = _.slice(shuffled, 0, testCount);
    const dataTrain = _.slice(shuffled, testCount);
    return [dataTest, dataTrain]
}


(async () => {
    const data = parser(rawData, { cast: true });
    const testSize = 50;
    const [dataTest, dataTrain] = splitDataSet(data, testSize)
    _.range(1, 20).forEach(k => {
        const result = _.chain(dataTest)
            .filter((item) => trainer(dataTrain, _.initial(item), k) === _.last(item))
            .size()
            .divide(testSize)
            .value()
        console.log(`k: ${k}, retult: ${result}`)
    })

})()

// CsvParserPromise(rawData)
//     .then(splitDataSet)
//     .then(trainer)
//     .then(console.log)

