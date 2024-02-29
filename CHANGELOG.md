# Changelog

All changes that impact users of this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

<!---
This document is intended for users of the applications and API. Changes to things
like tests should not be noted in this document.

When updating this file for a PR, add an entry for your change under Unreleased
and one of the following headings:
 - Added - for new features.
 - Changed - for changes in existing functionality.
 - Deprecated - for soon-to-be removed features.
 - Removed - for now removed features.
 - Fixed - for any bug fixes.
 - Security - in case of vulnerabilities.

If the heading does not yet exist under Unreleased, then add it as a 3rd heading,
with three #.


When preparing for a public release candidate add a new 2nd heading, with two #, under
Unreleased with the version number and the release date, in year-month-day
format. Then, add a link for the new version at the bottom of this document and
update the Unreleased link so that it compares against the latest release tag.


When preparing for a bug fix release create a new 2nd heading above the Fixed
heading to indicate that only the bug fixes and security fixes are in the bug fix
release.
-->

## [Unreleased]
### Added
- Apollo Metric drivers, tests, and data [#533](https://github.com/DOI-USGS/ale/pull/533)

### Fixed

- Fixed landed sensors to correctly project locally [#590](https://github.com/DOI-USGS/ale/pull/590)
- Fixed Hayabusa amica center time computation to match ISIS [#592](https://github.com/DOI-USGS/ale/pull/592)

## [0.10.0] - 2024-01-08 

### Added

- Mariner10 IsisLabelNaifSpice driver, tests, and test data [#547](https://github.com/DOI-USGS/ale/pull/547)
- Clementine UVVIS IsisLabelNaifSpice driver, tests, and test data [#546](https://github.com/DOI-USGS/ale/pull/546)
- Clementine LWIR, NIR, and HIRES IsisLabelNaifSpice driver, tests, and test data [#565](https://github.com/DOI-USGS/ale/pull/565)
- Hayabusa NIRS IsisLabelNaifSpice driver, tests and test data [#532](https://github.com/DOI-USGS/ale/pull/532)
- LO HighCamera IsisLabelNaifSpice driver, tests and test data [#533](https://github.com/DOI-USGS/ale/pull/553)
- DawnFC IsisLabelNaifSpice driver, tests, and test data [#567](https://github.com/DOI-USGS/ale/pull/567)
- DawnVIR IsisLabelNaifSpice driver, tests, and test data [#566](https://github.com/DOI-USGS/ale/pull/566)
- Added Local option to isd_generate, used to generate sky based projections for rover data [#564](https://github.com/DOI-USGS/ale/pull/564)
- Added Nadir option to isd_generate, used to generate nadir instrument pointing [#564](https://github.com/DOI-USGS/ale/pull/564)
- Added radial and tangential distortion model [#575](https://github.com/DOI-USGS/ale/pull/575)
- Updated the spiceypy version used in environment.yml [#552](https://github.com/DOI-USGS/ale/issues/552)

### Fixed
- Fixed LRO MiniRF drivers naif keywords focal to pixel and pixel to focal translations to be correct. [#569](https://github.com/DOI-USGS/ale/pull/569)
- Bugfix for position and orientation for MSL cameras (driver MslMastcamPds3NaifSpiceDriver). Validated that Nav and Mast LBL files (for both left and right sensor) produce correctly positioned and oriented CSM cameras, that are self-consistent and consistent with a prior DEM for the site. [#580](https://github.com/DOI-USGS/ale/pull/580) 
- Bug fix for ray direction for MSL. [#589](https://github.com/DOI-USGS/ale/pull/589)

### Changed
- Removed the affine6p library and replaced affine6p's affine transformation with a numpy solution [#579](https://github.com/DOI-USGS/ale/pull/579) 

## [0.9.1] - 2023-06-05

### Changed
- The NaifSpice class now gets two sun positions/velocities when a driver has more than one ephemeris time [#542](https://github.com/DOI-USGS/ale/pull/542)

### Fixed
- MexHrscIsisLabelNaifSpice and MexHrscPds3NaifSpice have had there ephemeris times changed and sampling factor updated. MexHrscIsisLabelNaifSpice has also had it's focal length, and focal plane translation updated to reflect those found in the MexHrscPds3NaifSpice driver [#541](https://github.com/DOI-USGS/ale/pull/541)
- MGS drivers now account for a time bias in the ephemeris data [#538](https://github.com/DOI-USGS/ale/pull/538)

## [0.9.0] - 2023-04-19

### Fixed
- Kaguya IsisLabelIsisSpice now calculates the right exposure_duration and focal2pixel_lines [#487](https://github.com/DOI-USGS/ale/pull/487)
- Logging from generate_isd now correctly limits logging information [#487](https://github.com/DOI-USGS/ale/pull/487)

### Changed
- Projection information is only written to the ISD if a projection is present instead of writing an empty projection [#528](https://github.com/DOI-USGS/ale/pull/528/)
- Disabled MSI drivers until tests are added [#526](https://github.com/DOI-USGS/ale/pull/526/)

### Added
- Projection information (through GDAL) will be attached to the ISD if a projected product is processed through ALE [#524](https://github.com/DOI-USGS/ale/pull/524)
- Kaguya IsisLabelNaifSpice driver, tests, and test data [#487](https://github.com/DOI-USGS/ale/pull/487)
- Hayabusa Amica IsisLabelNaifSpice driver, tests and test data [#521](https://github.com/DOI-USGS/ale/pull/521)
- Msi IsisLabelNaifSpice Driver [#511](https://github.com/DOI-USGS/ale/pull/511)
- MGS MOC WAC IsisLabelNaifSpice driver, tests, and test data [#516](https://github.com/DOI-USGS/ale/pull/516)
- Chandrayaan1_mrffr IsisLabelNaifSpice driver, tests and test data [#519](https://github.com/DOI-USGS/ale/pull/519)
- MGS MOC Narrow Angle IsisLabelNaifSpice driver, tests, and test data [#517](https://github.com/DOI-USGS/ale/pull/517)
- Hayabusa NIRS IsisLabelNaifSpice driver, tests and test data [#532](https://github.com/DOI-USGS/ale/pull/532)
<<<<<<< HEAD



=======
>>>>>>> 2c73f7da6ab8efbe056fff9c160e4458fec2f982
