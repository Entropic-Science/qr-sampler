"""``qr-sampler info`` command.

Shows detailed information about a specific component profile.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import click

from qr_sampler.profiles.loader import ProfileLoader

if TYPE_CHECKING:
    from qr_sampler.profiles.schema import (
        AmplifierProfile,
        EngineProfile,
        EntropySourceProfile,
        PresetProfile,
        SamplerProfile,
    )


def _format_field(label: str, value: Any) -> str:
    """Format a label-value pair for display."""
    return f"  {label + ':':<24s} {value}"


def _print_engine(profile: EngineProfile) -> None:
    """Print detailed engine profile info."""
    click.echo(click.style(f"Engine: {profile.name}", bold=True))
    click.echo(_format_field("ID", profile.id))
    if profile.description:
        click.echo(_format_field("Description", profile.description))
    click.echo(_format_field("Adapter", profile.adapter))
    if profile.entry_point_group:
        click.echo(_format_field("Entry point group", profile.entry_point_group))
    if profile.platform.os:
        click.echo(_format_field("OS", ", ".join(profile.platform.os)))
    if profile.platform.accelerator:
        click.echo(_format_field("Accelerator", ", ".join(profile.platform.accelerator)))
    if profile.dependencies:
        click.echo(_format_field("Dependencies", ", ".join(profile.dependencies)))
    if profile.docker.base_image:
        click.echo(_format_field("Docker image", profile.docker.base_image))
    if profile.docker.dockerfile:
        click.echo(_format_field("Dockerfile", profile.docker.dockerfile))
    if profile.defaults:
        click.echo("  Defaults:")
        for k, v in sorted(profile.defaults.items()):
            click.echo(f"    {k}: {v}")
    if profile.known_working_models:
        click.echo("  Known-working models:")
        for m in profile.known_working_models:
            click.echo(f"    - {m}")
    if profile.known_incompatible_models:
        click.echo("  Known-incompatible models:")
        for m in profile.known_incompatible_models:
            click.echo(f"    - {m}")


def _print_entropy(profile: EntropySourceProfile) -> None:
    """Print detailed entropy source profile info."""
    click.echo(click.style(f"Entropy source: {profile.name}", bold=True))
    click.echo(_format_field("ID", profile.id))
    if profile.description:
        click.echo(_format_field("Description", profile.description))
    click.echo(_format_field("Source class", profile.source_class))
    if profile.transport:
        click.echo(_format_field("Transport", profile.transport))
    if profile.dependencies:
        click.echo(_format_field("Dependencies", ", ".join(profile.dependencies)))
    if profile.compatible_amplifiers:
        click.echo("  Compatible amplifiers:")
        for a in profile.compatible_amplifiers:
            click.echo(f"    - {a}")
    if profile.known_incompatible_amplifiers:
        click.echo("  Known-incompatible amplifiers:")
        for a in profile.known_incompatible_amplifiers:
            click.echo(f"    - {a}")
    if profile.defaults:
        click.echo("  Defaults:")
        for k, v in sorted(profile.defaults.items()):
            click.echo(f"    {k}: {v}")


def _print_amplifier(profile: AmplifierProfile) -> None:
    """Print detailed amplifier profile info."""
    click.echo(click.style(f"Amplifier: {profile.name}", bold=True))
    click.echo(_format_field("ID", profile.id))
    if profile.description:
        click.echo(_format_field("Description", profile.description))
    click.echo(_format_field("Amplifier class", profile.amplifier_class))
    if profile.input_assumptions:
        click.echo(_format_field("Input assumptions", profile.input_assumptions))
    if profile.compatible_entropy_sources:
        click.echo("  Compatible entropy sources:")
        for s in profile.compatible_entropy_sources:
            click.echo(f"    - {s}")
    if profile.known_incompatible_entropy_sources:
        click.echo("  Known-incompatible sources:")
        for s in profile.known_incompatible_entropy_sources:
            click.echo(f"    - {s}")


def _print_sampler(profile: SamplerProfile) -> None:
    """Print detailed sampler profile info."""
    click.echo(click.style(f"Sampler: {profile.name}", bold=True))
    click.echo(_format_field("ID", profile.id))
    if profile.description:
        click.echo(_format_field("Description", profile.description))
    click.echo(_format_field("Sampler class", profile.sampler_class))
    click.echo(_format_field("Performance impact", profile.performance_impact))
    click.echo(_format_field("Build-time locked", str(profile.build_time_locked)))
    click.echo(_format_field("Requires vocab_size", str(profile.requires_vocab_size)))


def _print_preset(profile: PresetProfile) -> None:
    """Print detailed preset profile info."""
    label = f"Preset: {profile.name}"
    if profile.experimental:
        label += " [experimental]"
    click.echo(click.style(label, bold=True))
    click.echo(_format_field("ID", profile.id))
    if profile.origin:
        click.echo(_format_field("Origin", profile.origin))
    if profile.description:
        click.echo(_format_field("Description", profile.description))
    if profile.overrides:
        click.echo("  Overrides:")
        for k, v in sorted(profile.overrides.items()):
            click.echo(f"    {k}: {v}")


@click.command()
@click.argument(
    "component_type",
    type=click.Choice(
        ["engine", "entropy", "amplifier", "sampler", "preset"],
        case_sensitive=False,
    ),
)
@click.argument("name")
def info(component_type: str, name: str) -> None:
    """Show detailed info about a component profile.

    COMPONENT_TYPE is one of: engine, entropy, amplifier, sampler, preset.
    NAME is the profile identifier (e.g. vllm, quantum_grpc, creative_sampling).
    """
    loader = ProfileLoader()

    try:
        if component_type == "engine":
            _print_engine(loader.load_engine(name))
        elif component_type == "entropy":
            _print_entropy(loader.load_entropy_source(name))
        elif component_type == "amplifier":
            _print_amplifier(loader.load_amplifier(name))
        elif component_type == "sampler":
            _print_sampler(loader.load_sampler(name))
        elif component_type == "preset":
            _print_preset(loader.load_preset(name))
    except KeyError as exc:
        raise click.ClickException(str(exc)) from exc
