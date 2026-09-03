# 📊 Dataset

This directory contains the dataset and reproducibility files used in the
machine-learning analysis of inorganic phosphor emission wavelengths.

## 🔬 Main Dataset

**File:** `group_one_dopant_without_exact_duplicates.csv`

The dataset contains inorganic phosphor materials together with
physicochemical descriptors used as input variables for machine-learning
models.

**Target variable:** `Emission max. (nm)`

The dataset represents the single-dopant subset after removal of exact
duplicate entries.

## 📋 Descriptor Definitions

## Descriptor Definitions

The processed dataset contains material metadata, experimental information, the emission-wavelength target, and physicochemical descriptors derived from elemental, ionic, and electronic properties.

For descriptors ending in `_sum`, `_max`, `_min`, and `_diff`:

- `_sum` = sum of the corresponding property across the encoded constituent species.
- `_max` = maximum value of the property among the encoded constituent species.
- `_min` = minimum value of the property among the encoded constituent species.
- `_diff` = difference between the maximum and minimum values (`max - min`).

Descriptors beginning with `avg_` represent average values calculated according to the descriptor-generation procedure.

| Descriptor | Category | Description | Unit |
|---|---|---|---|
| `Inorganic phosphor` | Metadata | Name or chemical formula identifying the inorganic phosphor material. | — |
| `Host` | Metadata | Chemical composition or formula of the host lattice containing the luminescent dopant. | — |
| `1st dopant` | Dopant information | Identity or chemical symbol of the primary luminescent dopant/activator. | — |
| `1st dopant valency` | Dopant descriptor | Oxidation state (formal ionic charge) assigned to the primary dopant. | Dimensionless |
| `1st doping concentration` | Experimental | Reported concentration of the primary dopant. The concentration basis depends on the source-data convention. | Source-dependent |
| `Temp. (K)` | Experimental | Temperature associated with the reported optical measurement. | K |
| `Emission max. (nm)` | Target | Wavelength corresponding to the maximum intensity of the emission spectrum. This is the regression target. | nm |
| `Excitation source` | Experimental | Excitation wavelength used to stimulate photoluminescence. | nm |
| `Reference` | Metadata | Bibliographic reference associated with the phosphor record. | — |
| `MP-ID` | Identifier | Materials Project identifier for the corresponding material or structure, when available. | — |
| `ICSD-ID` | Identifier | Inorganic Crystal Structure Database identifier for the corresponding structure, when available. | — |
| `Atomic Ratio_max` | Composition descriptor | Maximum encoded atomic/stoichiometric ratio among the constituent species. | Dimensionless |
| `Atomic Ratio_min` | Composition descriptor | Minimum encoded atomic/stoichiometric ratio among the constituent species. | Dimensionless |
| `Atomic Ratio_diff` | Composition descriptor | Difference between the maximum and minimum encoded atomic ratios. | Dimensionless |
| `Atomic Number_sum` | Elemental descriptor | Sum of atomic numbers of the encoded constituent elements. | Dimensionless |
| `Atomic Number_max` | Elemental descriptor | Maximum atomic number among the encoded constituent elements. | Dimensionless |
| `Atomic Number_min` | Elemental descriptor | Minimum atomic number among the encoded constituent elements. | Dimensionless |
| `Atomic Number_diff` | Elemental descriptor | Difference between the maximum and minimum atomic numbers. | Dimensionless |
| `Atomic Weight_sum` | Elemental descriptor | Sum of atomic weights of the encoded constituent elements. | Verify encoding |
| `Atomic Weight_max` | Elemental descriptor | Maximum atomic weight among the encoded constituent elements. | Verify encoding |
| `Atomic Weight_min` | Elemental descriptor | Minimum atomic weight among the encoded constituent elements. | Verify encoding |
| `Atomic Weight_diff` | Elemental descriptor | Difference between the maximum and minimum atomic weights. | Verify encoding |
| `Atomic Radius_sum` | Elemental descriptor | Sum of atomic-radius values of the encoded constituent elements. | Verify source units |
| `Atomic Radius_max` | Elemental descriptor | Maximum atomic radius among the encoded constituent elements. | Verify source units |
| `Atomic Radius_min` | Elemental descriptor | Minimum atomic radius among the encoded constituent elements. | Verify source units |
| `Atomic Radius_diff` | Elemental descriptor | Difference between the maximum and minimum atomic radii. | Verify source units |
| `EN pauling_sum` | Electronegativity descriptor | Sum of Pauling electronegativity values of the encoded constituent elements. | Dimensionless |
| `EN pauling_max` | Electronegativity descriptor | Maximum Pauling electronegativity among the encoded constituent elements. | Dimensionless |
| `EN pauling_min` | Electronegativity descriptor | Minimum Pauling electronegativity among the encoded constituent elements. | Dimensionless |
| `EN pauling_diff` | Electronegativity descriptor | Difference between the maximum and minimum Pauling electronegativities. | Dimensionless |
| `Valence Electron_sum` | Electronic descriptor | Sum of encoded valence-electron counts of the constituent elements. | electrons |
| `Valence Electron_max` | Electronic descriptor | Maximum encoded valence-electron count among the constituent elements. | electrons |
| `Valence Electron_min` | Electronic descriptor | Minimum encoded valence-electron count among the constituent elements. | electrons |
| `Valence Electron_diff` | Electronic descriptor | Difference between the maximum and minimum encoded valence-electron counts. | electrons |
| `Ionization Energy_sum` | Electronic descriptor | Sum of ionization-energy values of the encoded constituent elements. | Verify source units |
| `Ionization Energy_max` | Electronic descriptor | Maximum ionization energy among the encoded constituent elements. | Verify source units |
| `Ionization Energy_min` | Electronic descriptor | Minimum ionization energy among the encoded constituent elements. | Verify source units |
| `Ionization Energy_diff` | Electronic descriptor | Difference between the maximum and minimum ionization energies. | Verify source units |
| `electronegativity_diff_ratio` | Engineered descriptor | Normalized or relative electronegativity-difference descriptor between relevant constituent species. Exact formula should be verified from the descriptor-generation code. | Formula-dependent |
| `radius_mismatch` | Engineered descriptor | Descriptor representing the degree of ionic or atomic radius mismatch between relevant species. Exact formula should be verified from the descriptor-generation code. | Formula-dependent |
| `electron_density` | Engineered descriptor | Electronic/compositional descriptor representing an electron-density-related quantity. Exact formula and units should be verified from the descriptor-generation code. | Formula-dependent |
| `mass_to_charge` | Engineered descriptor | Descriptor relating an encoded mass or atomic-weight quantity to charge/valence. Exact formula and units should be verified from the descriptor-generation code. | Formula-dependent |
| `ionic_radius_emission_center` | Ionic descriptor | Ionic radius assigned to the luminescent emission-center ion (primary dopant/activator) in the relevant ionic representation. | Verify source units |
| `EN_ligand_avg` | Chemical descriptor | Average electronegativity of ligand/anionic species associated with the luminescent center according to the descriptor-generation procedure. | Dimensionless |
| `ionic_radius_substituted` | Ionic descriptor | Ionic radius assigned to the host ion/site considered to be substituted by the luminescent dopant. | Verify source units |
| `avg_ionic_radius_host` | Host descriptor | Average ionic radius of the constituent species representing the host lattice. | Verify source units |
| `avg_EN_host` | Host descriptor | Average electronegativity of the constituent elements/species in the host lattice. | Dimensionless |
| `avg_atomic_number_host` | Host descriptor | Average atomic number of the constituent elements in the host lattice. | Dimensionless |
| `avg_group_host` | Host descriptor | Average periodic-table group number of the constituent host elements according to the elemental-property encoding. | Dimensionless |
| `avg_s_electrons` | Electronic descriptor | Average number of electrons assigned to s orbitals in the encoded electronic configurations. | electrons |
| `avg_p_electrons` | Electronic descriptor | Average number of electrons assigned to p orbitals in the encoded electronic configurations. | electrons |
| `avg_d_electrons` | Electronic descriptor | Average number of electrons assigned to d orbitals in the encoded electronic configurations. | electrons |
| `avg_f_electrons` | Electronic descriptor | Average number of electrons assigned to f orbitals in the encoded electronic configurations. | electrons |

### Notes

- `Emission max. (nm)` is the target variable and is not used as a predictor.
- `Inorganic phosphor`, `Host`, `1st dopant`, `Reference`, `MP-ID`, and `ICSD-ID` are metadata/identifier fields and are excluded from the numerical feature matrix in the main machine-learning workflow.
- The definitions above describe the physical meaning of the dataset columns. Exact mathematical formulas, weighting conventions, and units for dataset-specific engineered descriptors should be verified from the descriptor-generation code before being reported as formal definitions.
- The original optical-property data are derived from the **Optical Property Database of Inorganic Phosphor (IPOP)** dataset.
