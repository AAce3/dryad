use std::env;

use clap::{Arg, Command};
use uci::uci_loop;

pub mod mcts;
pub mod movegen;
pub mod nn;
pub mod uci;

pub const VERSION: &str = env!("CARGO_PKG_VERSION");

fn main() {
    let matches = Command::new("dryad")
        .version(VERSION)
        .author("Aaron Li")
        .arg(
            Arg::new("network-path")
                .long("network-path")
                .value_name("PATH")
                .help("Specifies the network path"),
        )
        .get_matches();

    println!("dryad {VERSION}");

    uci_loop(
        matches
            .get_one::<String>("network-path")
            .map(|x| x.as_str()),
    );
}
