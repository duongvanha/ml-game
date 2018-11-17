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


function normalization(data, countColumn) {
    const dataClone = _.cloneDeep(data);
    for (let i = 0; i < countColumn; i++) {
        const column = dataClone.map(row => row[i]);
        const max = _.max(column);
        const min = _.min(column);

        for (let j = 0; j < column.length; j++) {
            // if min == max infinity
            dataClone[j][i] = (dataClone[j][i] - min) / (max - min)

        }
    }

    return dataClone
}

const data = parser(rawData, { cast: true });
const testSize = 100;
const k = 10;
_.range(0, 3).forEach(feature => {
    const newData = data.map(row => [row[feature], _.last(row)]);
    const [dataTest, dataTrain] = splitDataSet(normalization(newData,1), testSize);

    const result = _.chain(dataTest)
        .filter((item) => trainer(dataTrain, _.initial(item), k) === _.last(item))
        .size()
        .divide(testSize)
        .value();
    console.log(`feature: ${feature}, retult: ${result}`)
});

// CsvParserPromise(rawData)
//     .then(splitDataSet)
//     .then(trainer)
//     .then(console.log)

