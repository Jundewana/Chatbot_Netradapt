import * as gk from '../src/genkit'
import { expect } from 'chai'
import 'mocha'
import * as gklib from './lib';

describe('Basic API', () => {
    let evalCostWrapperFn = (str: string[]): number => {
        return gklib.evalFns.cost(str, 'target'.split(''));
    }

    before( () => {
    });

    it('should generate random population - cost based', () => {
        let gaconfig: gk.Config = {
            dnaCodes: gklib.dnaCodes,
            evalType: gk.EvalType.ET_COST,
            evalFn: gklib.evalFns.cost,
            mutate: gklib.mutateFns.updown,
            mate: gklib.mateFns["half-half"]
        };

        let p = gk.randomPopulation(gaconfig, 10, 5);
        expect(p).to.have.property('elements');
        expect(p).to.have.property('generation');
        expect(p).property('elements').to.be.an('array');
        expect(p).property('generation').to.equal(0);
        expect(p.elements[0]).to.have.property('code');
        expect(p.elements[0].code).to.be.an('array');
        expect(p.elements[0]).to.have.property('cost');
        expect(p.elements[0]).to.have.property('fitness');
        expect(p.elements[0]).to.have.property('score');
    });

    it('should score population - cost based', () => {
        let gaconfig: gk.Config = {
            dnaCodes: gklib.dnaCodes,
            evalType: gk.EvalType.ET_COST,
            evalFn: evalCostWrapperFn,
            mutate: gklib.mutateFns.updown,
            mate: gklib.mateFns["half-half"]
        };

        let p = gk.randomPopulation(gaconfig, 6, 5);
        let p_scored = gk.score(gaconfig, p);
        expect(p_scored).to.have.property('elements');
        expect(p_scored).to.have.property('generation');
        expect(p_scored).property('elements').to.be.an('array');
        expect(p_scored).property('generation').to.equal(0);
        expect(p_scored.elements[0]).to.have.property('code');
        expect(p_scored.elements[0].code).to.be.an('array');
        expect(p_scored.elements[0]).to.have.property('cost');
        expect(p_scored.elements[0]).to.have.property('fitness');
        expect(p_scored.elements[0]).to.have.property('score');
        expect(p_scored.elements[0].fitness).to.be.a('undefined');
        expect(p_scored.elements[0].cost).not.to.be.a('undefined');
    });

    it('should evolve population - cost based', () => {
        let gaconfig: gk.Config = {
            dnaCodes: gklib.dnaCodes,
            evalType: gk.EvalType.ET_COST,
            evalFn: evalCostWrapperFn,
            mutate: gklib.mutateFns.updown,
            mate: gklib.mateFns["half-half"]
        };

        let p = gk.randomPopulation(gaconfig, 6, 5);
        let p_scored = gk.score(gaconfig, p);
        let p_next = gk.evolve(gaconfig, p_scored, 0.01);
        expect(p_next).to.have.property('elements');
        expect(p_next).to.have.property('generation');
        expect(p_next).property('elements').to.be.an('array');
        expect(p_next).property('generation').to.equal(1);
        expect(p_next.elements[0]).to.have.property('code');
        expect(p_next.elements[0].code).to.be.an('array');
        expect(p_next.elements[0]).to.have.property('cost');
        expect(p_next.elements[0]).to.have.property('fitness');
        expect(p_next.elements[0]).to.have.property('score');
        expect(p_next.elements[0].fitness).to.be.a('undefined');
        expect(p_next.elements[0].cost).not.to.be.a('undefined');
    });

    it('should generate random population - fitness based', () => {
        let gaconfig: gk.Config = {
            dnaCodes: gklib.dnaCodes,
            evalType: gk.EvalType.ET_FITNESS,
            evalFn: gklib.evalFns.fitness,
            mutate: gklib.mutateFns.random,
            mate: gklib.mateFns["half-half"]
        };

        let p = gk.randomPopulation(gaconfig, 10, 5);
        expect(p).to.have.property('elements');
        expect(p).to.have.property('generation');
        expect(p).property('elements').to.be.an('array');
        expect(p).property('generation').to.equal(0);
        expect(p.elements[0]).to.have.property('code');
        expect(p.elements[0].code).to.be.an('array');
        expect(p.elements[0]).to.have.property('cost');
        expect(p.elements[0]).to.have.property('fitness');
        expect(p.elements[0]).to.have.property('score');
    });

    it('should score population - fitness based', () => {
        let gaconfig: gk.Config = {
            dnaCodes: gklib.dnaCodes,
            evalType: gk.EvalType.ET_FITNESS,
            evalFn: gklib.evalFns.fitness,
            mutate: gklib.mutateFns.random,
            mate: gklib.mateFns["half-half"]
        };

        let p = gk.randomPopulation(gaconfig, 6, 5);
        let p_scored = gk.score(gaconfig, p);
        expect(p_scored).to.have.property('elements');
        expect(p_scored).to.have.property('generation');
        expect(p_scored).property('elements').to.be.an('array');
        expect(p_scored).property('generation').to.equal(0);
        expect(p_scored.elements[0]).to.have.property('code');
        expect(p_scored.elements[0].code).to.be.an('array');
        expect(p_scored.elements[0]).to.have.property('cost');
        expect(p_scored.elements[0]).to.have.property('fitness');
        expect(p_scored.elements[0]).to.have.property('score');
        expect(p_scored.elements[0].fitness).not.to.be.a('undefined');
        expect(p_scored.elements[0].cost).to.be.a('undefined');
    });

    it('should evolve population - fitness based', () => {
        let gaconfig: gk.Config = {
            dnaCodes: gklib.dnaCodes,
            evalType: gk.EvalType.ET_FITNESS,
            evalFn: gklib.evalFns.fitness,
            mutate: gklib.mutateFns.random,
            mate: gklib.mateFns["half-half"]
        };

        let p = gk.randomPopulation(gaconfig, 6, 5);
        let p_scored = gk.score(gaconfig, p);
        let p_next = gk.evolve(gaconfig, p_scored, 0.01);
        expect(p_next).to.have.property('elements');
        expect(p_next).to.have.property('generation');
        expect(p_next).property('elements').to.be.an('array');
        expect(p_next).property('generation').to.equal(1);
        expect(p_next.elements[0]).to.have.property('code');
        expect(p_next.elements[0].code).to.be.an('array');
        expect(p_next.elements[0]).to.have.property('cost');
        expect(p_next.elements[0]).to.have.property('fitness');
        expect(p_next.elements[0]).to.have.property('score');
        expect(p_next.elements[0].fitness).not.to.be.a('undefined');
        expect(p_next.elements[0].cost).to.be.a('undefined');
    });

    after( () => {
    });
})
