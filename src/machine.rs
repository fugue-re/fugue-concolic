use std::collections::BinaryHeap;
use std::sync::Arc;

use fugue::bytes::Order;
use fugue::ir::{AddressValue, IntoAddress, Translator};
use fugue::ir::il::Location;

use fuguex::machine::{Branch, Machine};
use fuguex::machine::types::{Bound, StepOutcome};
use fuguex::state::pcode::PCodeState;

use parking_lot::Mutex;

use crate::interpreter::{ConcolicContext, Error as ConcolicError};
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
struct StatePriority<P: Ord, O: Order>(P, Location, ConcolicState<O>, Bound<AddressValue>);

impl<P: Ord, O: Order> PartialEq for StatePriority<P, O> {
    fn eq(&self, other: &Self) -> bool {
        self.0.eq(&other.0)
    }
}

impl<P: Ord, O: Order> Eq for StatePriority<P, O> { }

impl<P: Ord, O: Order> PartialOrd for StatePriority<P, O> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.0.partial_cmp(&other.0)
    }
}

impl<P: Ord, O: Order> Ord for StatePriority<P, O> {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0.cmp(&other.0)
    }
}

#[derive(Clone)]
pub struct ConcolicMachine<O: Order, const OPERAND_SIZE: usize> {
    machine: Machine<ConcolicContext<O, OPERAND_SIZE>>,
    states: BinaryHeap<StatePriority<usize, O>>,
    state_filters: Arc<Mutex<Vec<Box<dyn FnMut(&Location, &ConcolicState<O>) -> bool>>>>,
    state_rankers: Arc<Mutex<Vec<Box<dyn FnMut(&Location, &ConcolicState<O>) -> StateRank>>>>,
    translator: Arc<Translator>
}


impl<O: Order, const OPERAND_SIZE: usize> ConcolicMachine<O, OPERAND_SIZE> {
    pub fn new(translator: Arc<Translator>, state: PCodeState<u8, O>) -> Self {
        Self {
            translator: translator.clone(),
            machine: Machine::new(ConcolicContext::new(translator, state)),
            states: BinaryHeap::new(),
            state_filters: Arc::new(Mutex::new(Vec::new())),
            state_rankers: Arc::new(Mutex::new(Vec::new())),
        }
    }

    pub fn add_filter<F>(&mut self, f: F)
    where F: FnMut(&Location, &ConcolicState<O>) -> bool + 'static {
        self.state_filters.lock().push(Box::new(f));
    }

    pub fn add_ranker<F>(&mut self, f: F)
    where F: FnMut(&Location, &ConcolicState<O>) -> StateRank + 'static {
        self.state_rankers.lock().push(Box::new(f));
    }

    fn update_states(&mut self, states: Vec<(Branch, ConcolicState<O>)>, bound: &Bound<AddressValue>) -> Vec<(Location, ConcolicState<O>)> {
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
            self.states.push(StatePriority(score, location, state, bound.clone()))
        }

        reached
    }

    pub fn step_all(&mut self) -> Result<(Bound<AddressValue>, StepOutcome<Vec<(Location, ConcolicState<O>)>>), Error> {
        while let Some(StatePriority(_, location, state, bound)) = self.states.pop() {
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

    pub fn step_from<L, B>(&mut self, location: L) -> Result<StepOutcome<Vec<(Location, ConcolicState<O>)>>, Error>
    where L: Into<Location> {
        self.step_until(location, Bound::unbounded()).map(|(_, v)| v)
    }

    pub fn step_until<L, B>(&mut self, location: L, until: Bound<B>) -> Result<(Bound<AddressValue>, StepOutcome<Vec<(Location, ConcolicState<O>)>>), Error>
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
