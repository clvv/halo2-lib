#![allow(unused_imports)]
#![allow(unused_variables)]
use halo2_base::gates::builder::{GateCircuitBuilder, GateThreadBuilder};
use halo2_base::gates::flex_gate::{FlexGateConfig, GateChip, GateInstructions, GateStrategy};
use halo2_base::halo2_proofs::{
    arithmetic::Field,
    circuit::*,
    dev::MockProver,
    halo2curves::bn256::{Bn256, Fr, G1Affine},
    plonk::*,
    poly::kzg::multiopen::VerifierSHPLONK,
    poly::kzg::strategy::SingleStrategy,
    poly::kzg::{
        commitment::{KZGCommitmentScheme, ParamsKZG},
        multiopen::ProverSHPLONK,
    },
    transcript::{Blake2bRead, TranscriptReadBuffer},
    transcript::{Blake2bWrite, Challenge255, TranscriptWriterBuffer},
};
use halo2_base::utils::ScalarField;
use halo2_base::{
    Context,
    QuantumCell::{Existing, Witness},
    SKIP_FIRST_PASS,
};
use itertools::Itertools;
use rand::rngs::OsRng;
use std::marker::PhantomData;

use criterion::{criterion_group, criterion_main};
use criterion::{BenchmarkId, Criterion};

use pprof::criterion::{Output, PProfProfiler};
// Thanks to the example provided by @jebbow in his article
// https://www.jibbow.com/posts/criterion-flamegraphs/

use quote::quote;
use rust_format::{RustFmt, Formatter};

const K: u32 = 19;

fn inner_prod_bench<F: ScalarField>(ctx: &mut Context<F>, a: Vec<F>, b: Vec<F>) {
    assert_eq!(a.len(), b.len());
    let a = ctx.assign_witnesses(a);
    let b = ctx.assign_witnesses(b);

    let chip = GateChip::default();

    ctx.smart_map::<usize>(0..(1 << K) / 16 - 10, |ctx, _| {
        chip.inner_product(ctx, a.clone(), b.clone().into_iter().map(Existing));
    });
}

fn main() {
    let k = 10u32;
    // create circuit for keygen
    let mut builder = GateThreadBuilder::new(false);
    inner_prod_bench(builder.main(0), vec![Fr::one(); 5], vec![Fr::one(); 5]);

    // Compose the rust witness gen code
    let witness_gen_code = builder.main(0).compose_rust_witness_gen().clone();

    let full_config = builder.full_config(k as usize, Some(20));

    builder.config(k as usize, Some(20));
    let circuit = GateCircuitBuilder::mock(builder);

    // Wrap it in a bare file that simply runs the witness gen code
    let raw = format!("{}", quote! {
        use halo2_base::halo2_proofs::{
            arithmetic::Field,
            circuit::*,
            dev::MockProver,
            halo2curves::bn256::{Bn256, Fr, G1Affine},
            plonk::*,
            poly::kzg::multiopen::VerifierSHPLONK,
            poly::kzg::strategy::SingleStrategy,
            poly::kzg::{
                commitment::{KZGCommitmentScheme, ParamsKZG},
                multiopen::ProverSHPLONK,
            },
            transcript::{Blake2bRead, TranscriptReadBuffer},
            transcript::{Blake2bWrite, Challenge255, TranscriptWriterBuffer},
        };
        fn main() {
            let compute = #witness_gen_code;
            let wires: Vec<Fr> = compute();

            println!("{:?}", wires[wires.len()-1]);

            // let mut builder = GateThreadBuilder::new(true);
            // let a = (0..5).map(|_| Fr::random(OsRng)).collect_vec();
            // let b = (0..5).map(|_| Fr::random(OsRng)).collect_vec();
            // inner_prod_bench(builder.main(0), a, b);
            // let circuit = GateCircuitBuilder::prover(builder, break_points);

            // let mut transcript = Blake2bWrite::<_, _, Challenge255<_>>::init(vec![]);
            // create_proof::<
            //     KZGCommitmentScheme<Bn256>,
            //     ProverSHPLONK<'_, Bn256>,
            //     Challenge255<G1Affine>,
            //     _,
            //     Blake2bWrite<Vec<u8>, G1Affine, Challenge255<_>>,
            //     _,
            // >(&params, &pk, &[circuit], &[&[]], OsRng, &mut transcript)
            // .expect("prover should not fail");
        }
    });

    // Write it to `examples/circuit_runtime.rs`
    // let data = RustFmt::default().format_str(raw).unwrap();

    std::fs::write("examples/inner_product_runtime.rs", raw).expect("Unable to write file");

    std::fs::write("examples/inner_product_config.json", full_config.to_json()).expect("Unable to write the config json file");

    let params = ParamsKZG::<Bn256>::setup(k, OsRng);
    let vk = keygen_vk(&params, &circuit).expect("vk should not fail");
    let pk = keygen_pk(&params, vk, &circuit).expect("pk should not fail");

    // let break_points = circuit.break_points.take();

    // println!("{:?}", break_points);

    // let mut builder = GateThreadBuilder::new(true);
    // let a = (0..5).map(|_| Fr::random(OsRng)).collect_vec();
    // let b = (0..5).map(|_| Fr::random(OsRng)).collect_vec();
    // inner_prod_bench(builder.main(0), a, b);
    // let circuit = GateCircuitBuilder::prover(builder, break_points);

    // let mut transcript = Blake2bWrite::<_, _, Challenge255<_>>::init(vec![]);
    // create_proof::<
    //     KZGCommitmentScheme<Bn256>,
    //     ProverSHPLONK<'_, Bn256>,
    //     Challenge255<G1Affine>,
    //     _,
    //     Blake2bWrite<Vec<u8>, G1Affine, Challenge255<_>>,
    //     _,
    // >(&params, &pk, &[circuit], &[&[]], OsRng, &mut transcript)
    // .expect("prover should not fail");

    // let strategy = SingleStrategy::new(&params);
    // let proof = transcript.finalize();
    // let mut transcript = Blake2bRead::<_, _, Challenge255<_>>::init(&proof[..]);
    // verify_proof::<
    //     KZGCommitmentScheme<Bn256>,
    //     VerifierSHPLONK<'_, Bn256>,
    //     Challenge255<G1Affine>,
    //     Blake2bRead<&[u8], G1Affine, Challenge255<G1Affine>>,
    //     _,
    // >(&params, pk.get_vk(), strategy, &[&[]], &mut transcript)
    // .unwrap();
}
