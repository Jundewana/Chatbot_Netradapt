import * as _ from 'underscore';

// The dna codes for our application
export let dnaCodes: Array<string> = 'abcdefghijklmnopqrstuvwxyz          ABCDEFGHIJKLMNOPQRSTUVWXYZ'.split('');

// Get number representation of a dna code
let toNumber = (a: string): number => {
    return a.charCodeAt(0);
}

/** Distance between two dna codes
 */
export let distance = {
    /**  The square of their difference
     */
    'squared-error': (a: number, b: number): number => {
        return (a - b) * (a - b);
    },
    /**  Just their difference
     */
    'absolute': (a: number, b: number): number => {
        if (a > b) return (a - b);
        else return (b - a);
    },
    /** Either 1 or 0
     */
    'binary': (a: any, b: any): number => {
        return (a === b ? 1 : 0);
    }
};

export let evalFns = {
    /** Cost is sum of errors at each position
     */
    'cost': (str: Array<any>, target: Array<any>): number => {
        let zipped: Array<any> = _.zip(str, target);
        let res = _.foldl(zipped, function(a: number, e: any) {
            return a + (distance['squared-error'](toNumber(e[0]), toNumber(e[1])));
        }, 0);
        return res;
    },
    /** Fitness is ratio of correct codes vs total
     */
    'fitness': (str: Array<any>, target: Array<any>): number => {
        var zipped: Array<any> = _.zip(str, target);
        var sum = _.foldl(zipped, (a, e) => {
            return a + (distance['binary'](e[0], e[1]));
        }, 0);
        return sum / str.length;
    }
};

/** Mutate a given dnaCode
 */
export let mutateFns = {
    /** It just moves either up or down (randomly)
     *   in the dna codes array
     */
    'updown': (dnaCode: any, dnaCodes: Array<any>): any => {
        let direction = Math.random() > 0.5 ? 1 : -1;
        let dnaCodesIndex = dnaCodes.indexOf(dnaCode);
        dnaCodesIndex += direction;

        // wrap-around
        if (dnaCodesIndex < 0) {
            dnaCodesIndex = dnaCodes.length - 1;
        }
        if (dnaCodesIndex >= dnaCodes.length) {
            dnaCodesIndex = 0;
        }
        return dnaCodes[dnaCodesIndex];
    },
    /** It just assigns a new code randomly
     */
    'random': (dnaCode: any, dnaCodes: Array<any>): any => {
        let r: number = _.random(dnaCodes.length - 1);
        return dnaCodes[r];
    }
};

/** Mating function
 */
export let mateFns = {
    /** It just picks first half from first parent
     *   and second half from second parent
     */
    'half-half': (str1: Array<any>, str2: Array<any>): Array<any> => {
        // NOTE: str1 and str2 are of same lengths
        let mid = str1.length / 2;
        let part1 = _.take(str1, mid);
        let part2 = _.drop(str2, mid);
        return part1.concat(part2);
    }
};
