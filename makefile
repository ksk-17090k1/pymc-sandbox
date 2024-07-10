# --- poetry ---

pj_name = app

# at first
pipx-init:
	brew install pipx
	pipx ensurepath
	sudo pipx ensurepath --global
	@echo "✨close terminal and run 'make poetry.toml'✨"

# at second
pt-init:
	pipx install poetry
	poetry new ${pj_name}
	@echo "✨ if you don't need the package mode, add 'package-mode = false' in [tool.poetry] section. ✨"

pt-install:
	cd ${pj_name} && poetry install

# at third
pt-set:
	cd ${pj_name} \
	&& mkdir .venv \
	&& poetry add pydantic \
	&& poetry add pytest --group dev \
	&& poetry add mypy --group dev \
	&& poetry add ruff --group dev \
	&& poetry add ipykernel --group dev

pt-update:
	cd ${pj_name} \
	&& rm poetry.lock \
	&& poetry install
