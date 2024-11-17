       subroutine vumat(
C Read only -
     1  nblock, ndir, nshr, nstatev, nfieldv, nprops, jInfoArray,
     2  stepTime, totalTime, dtArray, cmname, coordMp, charLength,
     3  props, density, strainInc, relSpinInc,
     4  tempOld, stretchOld, defgradOld, fieldOld,
     3  stressOld, stateOld, enerInternOld, enerInelasOld,
     6  tempNew, stretchNew, defgradNew, fieldNew,
C Write only -
     5  stressNew, stateNew, enerInternNew, enerInelasNew )
C
      include 'vaba_param.inc'
C
C J2 Mises Plasticity with kinematic hardening for plane 
C strain case.
C Elastic predictor, radial corrector algorithm.
C
C The state variables are stored as:
C      STATE(*,1) = back stress component 11
C      STATE(*,2) = back stress component 22
C      STATE(*,3) = back stress component 33
C      STATE(*,4) = back stress component 12
C      STATE(*,5) = equivalent plastic strain
C
C
C All arrays dimensioned by (*) are not used in this algorithm
      dimension props(nprops), density(nblock),
     1  coordMp(nblock,*),
     2  charLength(*), dtArray(*), strainInc(nblock,ndir+nshr),
     3  relSpinInc(*), tempOld(*),
     4  stretchOld(*), defgradOld(*),
     5  fieldOld(*),  stressOld(nblock,ndir+nshr),
     6  stateOld(nblock,nstatev), enerInternOld(nblock),
     7  enerInelasOld(nblock), tempNew(*),
     8  stretchNew(*), defgradNew(*), fieldNew(*),
     9  stressNew(nblock,ndir+nshr), stateNew(nblock,nstatev),
     2  enerInternNew(nblock), enerInelasNew(nblock), jInfoArray(*)
C
      parameter (i_info_AnnealFlag = 1, 
     *     i_info_Intpt    = 2, ! Integration station number
     *     i_info_layer  = 3, ! Layer number
     *     i_info_kspt   = 4, ! Section point number in current layer
     *     i_info_effModDefn = 5, ! =1 if Bulk/ShearMod need to be defined
     *     i_info_ElemNumStartLoc   = 6) ! Start loc of user element number
C
      parameter( zero = 0., one = 1., two = 2., three = 3.,
     1  third = one/three, half = .5, twoThirds = two/three,
     2  threeHalfs = 1.5 )

C
      character*80 cmname
C
      pointer (ptrjElemNum, jElemNum)
      dimension jElemNum(nblock)
C
      lAnneal = jInfoArray(i_info_AnnealFlag) 
      iLayer = jInfoArray(i_info_layer)
      kspt   = jInfoArray(i_info_kspt)
      intPt  = jInfoArray(i_info_Intpt)
      iUpdateEffMod = jInfoArray(i_info_effModDefn)
      iElemNumStartLoc = jInfoArray(i_info_ElemNumStartLoc)
      ptrjElemNum = loc(jInfoArray(iElemNumStartLoc))
c      
      e     = props(1)
      xnu   = props(2)
      yield = props(3)
      hard  = props(4)
C
      twomu  = e / ( one + xnu )
      thremu = threeHalfs * twomu
      sixmu  = three * twomu
      alamda = twomu * ( e - twomu ) / ( sixmu - two * e )
      term   = one / ( twomu * ( one + hard/thremu ) )
      con1   = sqrt( twoThirds )
C
      do 100 i = 1,nblock
C
C Trial stress
        trace  = strainInc(i,1) + strainInc(i,2) + strainInc(i,3)
        sig1 = stressOld(i,1) + alamda*trace + twomu*strainInc(i,1)
        sig2 = stressOld(i,2) + alamda*trace + twomu*strainInc(i,2)
        sig3 = stressOld(i,3) + alamda*trace + twomu*strainInc(i,3)
        sig4 = stressOld(i,4)                + twomu*strainInc(i,4)
C
C Trial stress measured from the back stress
        s1 = sig1 - stateOld(i,1)
        s2 = sig2 - stateOld(i,2)
        s3 = sig3 - stateOld(i,3)
        s4 = sig4 - stateOld(i,4)
C
C Deviatoric part of trial stress measured from the back stress
        smean = third * ( s1 + s2 + s3 )
        ds1 = s1 - smean
        ds2 = s2 - smean
        ds3 = s3 - smean
C
C Magnitude of the deviatoric trial stress difference
        dsmag = sqrt( ds1**2 + ds2**2 + ds3**2 + 2.*s4**2 )
C
C Check for yield by determining the factor for plasticity,
C zero for elastic, one for yield
        radius = con1 * yield
        facyld = zero
        if(  dsmag - radius .ge. zero ) facyld = one
C
C Add a protective addition factor to prevent a divide by zero 
C when dsmag is zero. If dsmag is zero, we will not have exceeded
C the yield stress and facyld will be zero.
        dsmag  = dsmag + ( one - facyld )
C
C Calculated increment in gamma (this explicitly includes the 
C time step)
        diff   = dsmag - radius
        dgamma = facyld * term * diff
C
C Update equivalent plastic strain
        deqps  = con1 * dgamma
        stateNew(i,5) = stateOld(i,5) + deqps
C
C Divide dgamma by dsmag so that the deviatoric stresses are
C explicitly converted to tensors of unit magnitude in the
C following calculations
        dgamma = dgamma / dsmag
C
C Update back stress
        factor  = hard * dgamma * twoThirds
        stateNew(i,1) = stateOld(i,1) + factor * ds1
        stateNew(i,2) = stateOld(i,2) + factor * ds2
        stateNew(i,3) = stateOld(i,3) + factor * ds3
        stateNew(i,4) = stateOld(i,4) + factor *  s4
C
C Update the stress
        factor   = twomu * dgamma
        stressNew(i,1) = sig1 - factor * ds1
        stressNew(i,2) = sig2 - factor * ds2
        stressNew(i,3) = sig3 - factor * ds3
        stressNew(i,4) = sig4 - factor *  s4
C
C Update the specific internal energy -
        stressPower = half * (
     1    ( stressOld(i,1)+stressNew(i,1) )*strainInc(i,1)
     1    +     ( stressOld(i,2)+stressNew(i,2) )*strainInc(i,2)
     1    +     ( stressOld(i,3)+stressNew(i,3) )*strainInc(i,3)
     1    + two*( stressOld(i,4)+stressNew(i,4) )*strainInc(i,4) )
C
        enerInternNew(i) = enerInternOld(i)
     1    + stressPower / density(i)
C
C Update the dissipated inelastic specific energy -
        plasticWorkInc = dgamma * half * (
     1    ( stressOld(i,1)+stressNew(i,1) )*ds1
     1    +     ( stressOld(i,2)+stressNew(i,2) )*ds2
     1    +     ( stressOld(i,3)+stressNew(i,3) )*ds3
     1    + two*( stressOld(i,4)+stressNew(i,4) )*s4 )
        enerInelasNew(i) = enerInelasOld(i)
      1   + plasticWorkInc / density(i)
  100 continue
C
      return
      end