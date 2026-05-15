"""``qr-sampler list`` command group.

Provides subcommands for listing available engines, models, entropy
sources, amplifiers, and samplers from the profile system.
"""

from __future__ import annotations

import click

from qr_sampler.profiles.loader import ProfileLoader


@click.group("list")
def list_profiles() -> None:
    """List available components (engines, models, entropy sources, etc.)."""


@list_profiles.command()
def engines() -> None:
    """List available engine profiles."""
    loader = ProfileLoader()
    profiles = loader.list_engines()
    if not profiles:
        click.echo("No engine profiles found.")
        return
    for p in profiles:
        click.echo(f"  {p.id:<20s} {p.name}")
        if p.description:
            click.echo(f"  {'':<20s} {p.description}")


@list_profiles.command()
@click.option(
    "--engine",
    default=None,
    help="Filter to models known-working with this engine.",
)
def models(engine: str | None) -> None:
    """List known-working models."""
    loader = ProfileLoader()

    if engine is not None:
        try:
            profile = loader.load_engine(engine)
        except KeyError as exc:
            raise click.ClickException(str(exc)) from exc
        if not profile.known_working_models:
            click.echo(f"No known-working models for engine {engine!r}.")
            return
        click.echo(f"Known-working models for engine {engine!r}:")
        for m in profile.known_working_models:
            click.echo(f"  - {m}")
    else:
        # Show models from all engines, grouped.
        engine_profiles = loader.list_engines()
        for ep in engine_profiles:
            if ep.known_working_models:
                click.echo(f"{ep.id}:")
                for m in ep.known_working_models:
                    click.echo(f"  - {m}")


@list_profiles.command("entropy-sources")
def entropy_sources() -> None:
    """List available entropy source profiles."""
    loader = ProfileLoader()
    profiles = loader.list_entropy_sources()
    if not profiles:
        click.echo("No entropy source profiles found.")
        return
    for p in profiles:
        click.echo(f"  {p.id:<20s} {p.name}")
        if p.description:
            click.echo(f"  {'':<20s} {p.description}")


@list_profiles.command()
@click.option(
    "--entropy",
    "entropy_source",
    default=None,
    help="Filter to amplifiers compatible with this entropy source.",
)
def amplifiers(entropy_source: str | None) -> None:
    """List available amplifier profiles."""
    loader = ProfileLoader()

    if entropy_source is not None:
        try:
            source = loader.load_entropy_source(entropy_source)
        except KeyError as exc:
            raise click.ClickException(str(exc)) from exc
        if not source.compatible_amplifiers:
            click.echo(f"No compatible amplifiers for entropy source {entropy_source!r}.")
            return
        click.echo(f"Compatible amplifiers for entropy source {entropy_source!r}:")
        for a in source.compatible_amplifiers:
            click.echo(f"  - {a}")
    else:
        profiles = loader.list_amplifiers()
        if not profiles:
            click.echo("No amplifier profiles found.")
            return
        for p in profiles:
            click.echo(f"  {p.id:<20s} {p.name}")
            if p.description:
                click.echo(f"  {'':<20s} {p.description}")


@list_profiles.command()
def samplers() -> None:
    """List available sampler profiles."""
    loader = ProfileLoader()
    profiles = loader.list_samplers()
    if not profiles:
        click.echo("No sampler profiles found.")
        return
    for p in profiles:
        impact = f" (performance: {p.performance_impact})" if p.performance_impact != "none" else ""
        click.echo(f"  {p.id:<20s} {p.name}{impact}")
        if p.description:
            click.echo(f"  {'':<20s} {p.description}")


@list_profiles.command()
def presets() -> None:
    """List available sampling preset profiles."""
    loader = ProfileLoader()
    profiles = loader.list_presets()
    if not profiles:
        click.echo("No preset profiles found.")
        return
    for p in profiles:
        flag = " [experimental]" if p.experimental else ""
        click.echo(f"  {p.id:<20s} {p.name}{flag}")
        if p.origin:
            click.echo(f"  {'':<20s} origin: {p.origin}")
        if p.description:
            click.echo(f"  {'':<20s} {p.description}")
