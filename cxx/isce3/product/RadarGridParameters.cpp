#include "RadarGridParameters.h"

#include "RadarGridProduct.h"
#include <iomanip>   // <-- Add this

isce3::product::RadarGridParameters::
RadarGridParameters(const RadarGridProduct & product, char frequency) :
    RadarGridParameters(product.swath(frequency), product.lookSide())
{
    validate();
}

isce3::product::RadarGridParameters::
RadarGridParameters(const Swath & swath, isce3::core::LookSide lookSide) :
    _lookSide(lookSide),
    _sensingStart(swath.zeroDopplerTime()[0]),
    _wavelength(swath.processedWavelength()),
    _prf(1.0 / swath.zeroDopplerTimeSpacing()),
    _startingRange(swath.slantRange()[0]),
    _rangePixelSpacing(swath.rangePixelSpacing()),
    _rlength(swath.lines()),
    _rwidth(swath.samples()),
    _refEpoch(swath.refEpoch())
{
    // ---- Add debug prints BEFORE validate() ----
    std::cout << std::setprecision(16) << std::fixed;

    std::cout << "\n[DEBUG RadarGridParameters CTOR]\n";
    std::cout << "  lookSide          = " << static_cast<int>(_lookSide) << "\n";
    std::cout << "  sensingStart      = " << _sensingStart << "\n";
    std::cout << "  startingRange     = " << _startingRange << "\n";
    std::cout << "  wavelength        = " << _wavelength << "\n";
    std::cout << "  prf               = " << _prf << "\n";
    std::cout << "  rangePixelSpacing = " << _rangePixelSpacing << "\n";
    std::cout << "  rlength (lines)   = " << _rlength << "\n";
    std::cout << "  rwidth  (samples) = " << _rwidth  << "\n";
    std::cout << "  refEpoch          = " << _refEpoch << "\n";

    // Show derived quantities too
    std::cout << "  sensingStop()     = " << sensingStop() << "\n";
    std::cout << "  endingRange()     = " << endingRange() << "\n";
    std::cout << "--------------------------------------\n";

    // ---- Now call validate() ----
    validate();
}

bool isce3::product::RadarGridParameters::
contains(const double aztime, const double srange) const {
    const auto halfAzimuthTimeInterval = azimuthTimeInterval() / 2;
    const auto halfRangePixelSpacing = rangePixelSpacing() / 2;
    return aztime >= _sensingStart - halfAzimuthTimeInterval
            and srange >= _startingRange - halfRangePixelSpacing
            and aztime <= sensingStop() + halfAzimuthTimeInterval
            and srange <= endingRange() + halfRangePixelSpacing;
}
