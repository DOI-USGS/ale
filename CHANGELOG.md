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

### Fixed
- Kaguya IsisLabelIsisSpice now calculates the right exposure_duration and focal2pixel_lines [#487](https://github.com/DOI-USGS/ale/pull/487)
- Logging from generate_isd now correctly limits logging information [#487](https://github.com/DOI-USGS/ale/pull/487)

### Changed
- Projection information is only written to the ISD if a projection is present instead of writing an empty projection [#528](https://github.com/DOI-USGS/ale/pull/528/)

### Added
- Projection information (through GDAL) will be attached to the ISD if a projected product is processed through ALE [#524](https://github.com/DOI-USGS/ale/pull/524)
- Kaguya IsisLabelNaifSpice driver, tests, and test data [#487](https://github.com/DOI-USGS/ale/pull/487)
- Hayabusa Amica IsisLabelNaifSpice driver, tests and test data [#521](https://github.com/DOI-USGS/ale/pull/521)
- Apollo Metric drivers, tests, and data [#533](https://github.com/DOI-USGS/ale/pull/533)
