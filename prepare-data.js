#!/usr/bin/env node
const fs = require('fs');
const path = require('path');

validateParameters();

const [inputDir, outputDir] = getDirectoriesFromCommandLine();

const files = fs.readdirSync(inputDir);

for (const file of files) {
    const src = path.join(inputDir, file);
    if (isDir(src)) {
        continue;
    }
    const txt = fs.readFileSync(src, 'utf-8')
        .replace(/\n/g, ' ')
        .replace(/\[[^]+?]/g, '')
        .replace(/[ \t]+/g, ' ').trim();

    const trg = path.join(outputDir, file);
    fs.writeFileSync(trg, txt);
}

function validateParameters() {
    if (process.argv.length !== 4) {
        console.log('usage: ./strip_bbcode.js inputdir outoutdir');
        process.exit(0);
    }

    process.argv.slice(2).forEach(dir => {
        const p = getAbsolutePath(dir);
        if (!exists(p)) {
            console.error(`Specified directory ${dir} does not exist`);
            process.exit(1);
        }
        if (!isDir(p)) {
            console.error(`Specified directory ${dir} is not a directory`);
            process.exit(1);
        }
    });
}

function getAbsolutePath(p) {
    return path.join(__dirname, p);
}

function getDirectoriesFromCommandLine() {
    return process.argv.slice(2).map(getAbsolutePath);
}

function exists(p) {
    return fs.existsSync(p);
}

function isDir(p) {
    return fs.statSync(p).isDirectory();
}
