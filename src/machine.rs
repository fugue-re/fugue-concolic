use std::collections::BinaryHeap;
use std::sync::Arc;
use std::marker::PhantomData;

use fugue::bytes::Order;
use fugue::ir::{AddressValue, IntoAddress, Translator};
use fugue::ir::il::Location;

use metaemu::loader::LoaderMapping;
use metaemu::machine::{Branch, Machine};
use metaemu::machine::types::{Bound, StepOutcome};
use metaemu::state::pcode::PCodeState;

use parking_lot::Mutex;

use crate::backend::ValueSolver;
use crate::interpreter::{ConcolicContext, Error as ConcolicError};
use crate::pointer::{DefaultPointerStrategy, SymbolicPointerStrategy};
use crate::state::ConcolicState;

use thiserror::Error;

#[derive(Debug, Error)]
pub enum Error {
    #[error("goal not reached")]
    GoalNotReached,
    #[error(transparent)]
    Machine(#[from] ConcolicError),
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum StateRank {
    GoalReached,
    Priority(usize),
}

#[derive(Clone)]
struct StatePriority<'ctx, P: Ord, O: Order, VS: ValueSolver<'ctx>>(P, Location, ConcolicState<'ctx, O, VS>, Bound<AddressValue>, PhantomData<&'ctx VS>);

impl<'ctx, P: Ord, O: Order, VS: ValueSolver<'ctx>> PartialEq for StatePriority<'ctx, P, O, VS> {
    fn eq(&self, other: &Self) -> bool {
        self.0.eq(&other.0)
    }
}

impl<'ctx, P: Ord, O: Order, VS: ValueSolver<'ctx>> Eq for StatePriority<'ctx, P, O, VS> { }

impl<'ctx, P: Ord, O: Order, VS: ValueSolver<'ctx>> PartialOrd for StatePriority<'ctx, P, O, VS> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.0.partial_cmp(&other.0)
    }
}

impl<'ctx, P: Ord, O: Order, VS: ValueSolver<'ctx>> Ord for StatePriority<'ctx, P, O, VS> {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0.cmp(&other.0)
    }
}

#[derive(Clone)]
pub struct ConcolicMachine<'ctx, O: Order, VS: ValueSolver<'ctx>, P: SymbolicPointerStrategy<'ctx, O, VS>, const OPERAND_SIZE: usize> {
    machine: Machine<ConcolicContext<'ctx, O, VS, P, OPERAND_SIZE>>,
    states: BinaryHeap<StatePriority<'ctx, usize, O, VS>>,
    state_filters: Arc<Mutex<Vec<Box<dyn FnMut(&Location, &ConcolicState<'ctx, O, VS>) -> bool + 'ctx>>>>,
    state_rankers: Arc<Mutex<Vec<Box<dyn FnMut(&Location, &ConcolicState<'ctx, O, VS>) -> StateRank + 'ctx>>>>,
    translator: Arc<Translator>
}

impl<'ctx, O: Order, VS: ValueSolver<'ctx>, const OPERAND_SIZE: usize> ConcolicMachine<'ctx, O, VS, DefaultPointerStrategy<'ctx, O, VS>, OPERAND_SIZE> {
    pub fn new(solver: VS, translator: Arc<Translator>, state: PCodeState<u8, O>) -> Self {
        Self {
            translator: translator.clone(),
            machine: Machine::new(ConcolicContext::new(solver, translator, state, DefaultPointerStrategy::default())),
            states: BinaryHeap::new(),
            state_filters: Arc::new(Mutex::new(Vec::new())),
            state_rankers: Arc::new(Mutex::new(Vec::new())),
        }
    }

    pub fn new_from(solver: VS, loader: impl LoaderMapping<PCodeState<u8, O>>) -> Self {
        let translator = loader.translator();
        Self::new_with(solver, translator, loader.into_state(), DefaultPointerStrategy::default())
    }
}

impl<'ctx, O: Order, VS: ValueSolver<'ctx>, P: SymbolicPointerStrategy<'ctx, O, VS>, const OPERAND_SIZE: usize> ConcolicMachine<'ctx, O, VS, P, OPERAND_SIZE> {
    pub fn new_with(solver: VS, translator: Arc<Translator>, state: PCodeState<u8, O>, pointer_strategy: P) -> Self {
        Self {
            translator: translator.clone(),
            machine: Machine::new(ConcolicContext::new(solver, translator, state, pointer_strategy)),
            states: BinaryHeap::new(),
            state_filters: Arc::new(Mutex::new(Vec::new())),
            state_rankers: Arc::new(Mutex::new(Vec::new())),
        }
    }

    pub fn new_from_with(solver: VS, loader: impl LoaderMapping<PCodeState<u8, O>>, pointer_strategy: P) -> Self {
        let translator = loader.translator();
        Self::new_with(solver, translator, loader.into_state(), pointer_strategy)
    }

    pub fn add_filter<F>(&mut self, f: F)
    where F: FnMut(&Location, &ConcolicState<'ctx, O, VS>) -> bool + 'ctx {
        self.state_filters.lock().push(Box::new(f));
    }

    pub fn add_ranker<F>(&mut self, f: F)
    where F: FnMut(&Location, &ConcolicState<'ctx, O, VS>) -> StateRank + 'ctx {
        self.state_rankers.lock().push(Box::new(f));
    }

    pub fn interpreter(&self) -> &ConcolicContext<'ctx, O, VS, P, OPERAND_SIZE> {
        self.machine.interpreter()
    }

    pub fn interpreter_mut(&mut self) -> &mut ConcolicContext<'ctx, O, VS, P, OPERAND_SIZE> {
        self.machine.interpreter_mut()
    }

    pub fn state(&self) -> &ConcolicState<'ctx, O, VS> {
        self.machine.interpreter().state()
    }

    pub fn state_mut(&mut self) -> &mut ConcolicState<'ctx, O, VS> {
        self.machine.interpreter_mut().state_mut()
    }

    fn update_states(&mut self, states: Vec<(Branch, ConcolicState<'ctx, O, VS>)>, bound: &Bound<AddressValue>) -> Vec<(Location, ConcolicState<'ctx, O, VS>)> {
        let mut reached = Vec::new();

        let mut filters = self.state_filters.lock();
        let mut rankers = self.state_rankers.lock();

        let step_state = self.machine.step_state();

        'outer: for (location, state) in states.into_iter().filter_map(|(b, s)| {
            let location = step_state.branch_location(b);
            if filters.iter_mut().any(|f| f(&location, &s)) {
                None
            } else {
                Some((location, s))
            }
        }) {
            let mut score = 0;
            for ranker in rankers.iter_mut() {
                match ranker(&location, &state) {
                    StateRank::GoalReached => {
                        reached.push((location, state));
                        continue 'outer
                    },
                    StateRank::Priority(p) => {
                        score += p;
                    }
                }
            }
            self.states.push(StatePriority(score, location, state, bound.clone(), PhantomData))
        }

        reached
    }

    pub fn step_all(&mut self) -> Result<(Bound<AddressValue>, StepOutcome<Vec<(Location, ConcolicState<'ctx, O, VS>)>>), Error> {
        while let Some(StatePriority(_, location, state, bound, _)) = self.states.pop() {
            self.machine.interpreter_mut().restore_state(state);
            let (nbound, outcome) = self.machine.step_until(location, bound)?;

            match outcome {
                StepOutcome::Reached => if !self.states.is_empty() {
                    continue
                } else {
                    return Ok((nbound, StepOutcome::Reached));
                },
                StepOutcome::Halt(states) => {
                    let goals = self.update_states(states, &nbound);
                    if goals.len() > 0 {
                        return Ok((nbound, StepOutcome::Halt(goals)));
                    }
                },
                _ => unreachable!("StepOutcome::Branch handled in Machine::step_until")
            }
        }

        Err(Error::GoalNotReached)
    }

    pub fn step_from<L>(&mut self, location: L) -> Result<StepOutcome<Vec<(Location, ConcolicState<'ctx, O, VS>)>>, Error>
    where L: Into<Location> {
        self.step_until(location, Bound::unbounded()).map(|(_, v)| v)
    }

    pub fn step_until<L, B>(&mut self, location: L, until: Bound<B>) -> Result<(Bound<AddressValue>, StepOutcome<Vec<(Location, ConcolicState<'ctx, O, VS>)>>), Error>
    where L: Into<Location>,
          B: IntoAddress {
        self.states.clear();

        let (bound, outcome) = self.machine.step_until(location, until)?;

        match outcome {
            StepOutcome::Reached => {
                return Ok((bound, StepOutcome::Reached));
            },
            StepOutcome::Halt(states) => {
                let goals = self.update_states(states, &bound);
                if goals.len() > 0 {
                    return Ok((bound, StepOutcome::Halt(goals)));
                }
            },
            _ => unreachable!("StepOutcome::Branch handled in Machine::step_until")
        }

        self.step_all()
    }
}
