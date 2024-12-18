# Genkit

A library for implementing genetic algorithm based applications.

## Installation

Install `genkit` using npm package manager

```sh
npm install --save genkit
```

## API Guide

### Config object

Create a `Config` object, which stores the context for rest of the operations.

```typescript
import * as gk from 'genkit';

let gaconfig: gk.Config = {
            dnaCodes: dnaCodes,
            evalType: evalType,
            evalFn: evalFn,
            mutate: mutateFn,
            mate: mateFn
};
```

`Config` is defined as

```typescript
type Config = {
    dnaCodes: Array<any>;
    evalType: EvalType;
    evalFn: Function;
    mutate: Function;
    mate: Function;
};
```

where

| Parameter  | Details                                                                                                                                                                                                                                                                                    |
| ---------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `dnaCodes` | Array of dna codes of any type, depending on what you are trying to model. Usually an `Array<string>` would be sufficient. For example, if you are trying to model the human genome, the `dnaCode` would be `['A', 'T', 'G', 'C']`. All dnas will be a sequence of elements of this array. |
| `evalType` | One of `EvalType.ET_COST`, `EvalType.ET_FITNESS`                                                                                                                                                                                                                                           |
| `evalFn`   | The evaluation function that takes a dna (sequence of dna codes) and returns either the cost or fitness of it depending on the `evalType`. It must be coupled with the above specified `evalType`. The type signature is `(str: Array<any>): number`                                       |
| `mutate`   | A function that takes a dna code (`any`) and gives out another dna code (`any`) by selecting from a given array of dna codes based on some logic. The type signature is `(dnaCode: any, dnaCodes: Array<any>): any`.                                                                       |
| `mate`     | A function that takes two dna sequences (parents) and generates an output dna sequence (child). The type signature is: `(str`: Array<any>, str2: Array<any>): Array<any>`                                                                                                                  |

### Generate initial population (random)

Once you have a `Config` object, you can start with creating a random population of a given dna sequence length and population count using the function `randomPopulation` as follows:

```typescript
let p = gk.randomPopulation(gaconfig, 10, 50);
```

It returns a `Population` object, which is defined as follows:

```typescript
type Population = {
    elements: Array<Gene>;
    generation: number;
};

```

It contains an array of `Gene` objects, which represents the dna sequence of each member of the population. The `Gene` object is defined as:

```typescript
type Gene = {
    code: Array<any>;
    score: number;
    fitness: number;
    cost: number;
};

```

### Evolve population

The next step would be to iteratively evolve the initial population towards more and more fitness or less and less cost at each step (depending on what you provided in the `evalType` parameter while creating the `Config`).

To calculate the score of any `Population` object, use the `score` function, which will return a new `Population` object that has its members scored according to the `evalType` provided in the `Config` object.

```typescript
p_scored = gk.score(gaconfig, p)
```

To get to the next step in evolution, call the `evolve` function on a scored `Population` object as follows:

```typescript
p_next = gk.evolve(gaconfig, p_scored, 0.01);
```

The last parameter in `evolve` function is the mutation rate. It is the probability of mutation happening during each mating operation. The higher the mutation rate, the more frequently the population can be expected to get random dna sequences which are not just derived from parents. Usually, low mutation rates work well because they provide enough opportunity for a given member to participate in natural selection.

You should apply some convergence checks after each step in evolution to find out it the current `Population` object is good enough for your purpose. If so, end the iterations. For more convinience, each `Gene` object also has a `score` member, which is a number betwee `0` and `1`, which represents a normalized "goodness" of that `Gene` object. So a convergence check can be simple done on the value of the `score`.

```typescript
// Using underscore.js
let best_gene = _.max(p_scored.elements, (e) => e.score);
```

That's it. For a working sample application, refer to this project: https://github.com/vaibhav276/genkit-examples/tree/master/target-text. (Most relevant code is in https://github.com/vaibhav276/genkit-examples/blob/master/target-text/src/app/ga-runner.service.ts)




