from dataclasses import dataclass, field
from typing import List, Tuple
from xml.etree.ElementTree import Element, SubElement
from copy import copy
from itertools import islice
from dataclasses_json import dataclass_json, config
from marshmallow import fields
from terminaltables import AsciiTable
from evrpscp import is_close
from math import isfinite, isnan

def encode_float(f: float, prec=2):
    if isfinite(f):
        return float(f)
    elif f > 0:
        return 'Infinity'
    else:
        return '-Infinity'

def decode_float(f: str):
    if f == 'Infinity':
        return float('inf')
    elif f == '-Infinity':
        return float('-inf')
    else:
        return float(f)

@dataclass_json
@dataclass
class PiecewiseLinearSegment:
    lowerBound: float = field(
        default=float('-inf'),
        metadata=config(
            encoder=encode_float,
            decoder=decode_float,
            mm_field=fields.Float()
        )
    )
    upperBound: float = field(
        default=float('inf'),
        metadata=config(
            encoder=encode_float,
            decoder=decode_float,
            mm_field=fields.Float()
        )
    )
    imageLowerBound: float = 0
    imageUpperBound: float = 0
    slope: float = 0

    def __post_init__(self):
        allows_arith = lambda x: isfinite(x) and not isnan(x)
        if allows_arith(self.upperBound) and allows_arith(self.lowerBound):
            assert is_close((self.upperBound - self.lowerBound) * self.slope, self.imageUpperBound - self.imageLowerBound), \
                f'Invalid piecewise linear segment: ({self.upperBound} - {self.lowerBound}) * {self.slope} ' \
                f'= {(self.upperBound - self.lowerBound) * self.slope}' \
                f' != {self.imageUpperBound - self.imageLowerBound} = {self.imageUpperBound} - {self.imageLowerBound} '

    #@property
    #def imageUpperBound(self):
    #    p = self.imageLowerBound + (self.upperBound - self.lowerBound) * self.slope
    #    return () if not isnan(p) else 0.0

    def inverse(self) -> 'PiecewiseLinearSegment':
        return PiecewiseLinearSegment(lowerBound=self.imageLowerBound, upperBound=self.imageUpperBound,
                                      imageLowerBound=self.lowerBound, imageUpperBound=self.upperBound,
                                      slope=1.0/self.slope)

    def with_slope(self, slope, scale_image=True) -> 'PiecewiseLinearSegment':
        if scale_image:
            return PiecewiseLinearSegment(lowerBound=self.lowerBound, upperBound=self.upperBound,
                                          imageLowerBound=self.imageLowerBound,
                                          imageUpperBound=self.imageLowerBound + (self.upperBound - self.lowerBound) * slope,
                                          slope=slope)
        else:
            return PiecewiseLinearSegment(imageLowerBound=self.imageLowerBound, imageUpperBound=self.imageUpperBound,
                                          lowerBound=self.lowerBound,
                                          upperBound=self.lowerBound + (self.imageUpperBound - self.imageLowerBound) / slope,
                                          slope=slope)

    def scale_slope(self, scale: float = 1., scale_image=True) -> 'PiecewiseLinearSegment':
        return self.with_slope(self.slope * scale, scale_image=scale_image)

    def at(self, x: float):
        if self.lowerBound - x > 0.01 or x - self.upperBound > 0.01:
            raise ValueError(f'{x} is not on the segment!')
        if self.slope != 0:
            return min(self.imageLowerBound + (x - self.lowerBound) * self.slope, self.imageUpperBound)
        else:
            return self.imageLowerBound

    def __str__(self):
        return f'S[{round(self.lowerBound, 2)}-{round(self.upperBound, 2)}|{round(self.imageLowerBound, 2)}-{round(self.imageUpperBound, 2)}|{round(self.slope, 2)}]'

    def __repr__(self):
        return str(self)

    def __eq__(self, other: 'PiecewiseLinearSegment'):
        return tuple(map(lambda x: round(x,2), (self.lowerBound, self.upperBound, self.imageLowerBound, self.slope))) \
               == tuple(map(lambda x: round(x,2), (other.lowerBound, other.upperBound, other.imageLowerBound, other.slope)))

    def __lt__(self, other: 'PiecewiseLinearSegment'):
        return self.lowerBound < other.lowerBound

    def __hash__(self):
        return id(self)

    @property
    def is_dummy(self) -> bool:
        """
        Returns true if the segment is a dummy segment, i.e. does not have properly defined bounds.
        """
        return self.lowerBound == float('-inf') or self.upperBound == float('inf')

    def toXML(self) -> Element:
        xml_segment = Element('Segment')

        xml_ub = SubElement(xml_segment, 'UpperBound')
        xml_ub.text = f'{self.upperBound:.6f}'

        xml_val = SubElement(xml_segment, 'Slope')
        xml_val.text = f'{self.slope:.6f}'
        return xml_segment

    @staticmethod
    def FromDomain(lower_bound: float, upper_bound: float, slope: float, image_lower_bound: float = 0.0) -> 'PiecewiseLinearSegment':
        return PiecewiseLinearSegment(lowerBound=lower_bound, upperBound=upper_bound, slope=slope,
                                      imageLowerBound=image_lower_bound, imageUpperBound=image_lower_bound + (upper_bound-lower_bound)*slope)

    @staticmethod
    def FromImage(image_lower_bound: float, image_upper_bound: float, slope: float, lower_bound: float = 0.0)\
            -> 'PiecewiseLinearSegment':
        return PiecewiseLinearSegment(lowerBound=lower_bound,
                                      upperBound=lower_bound + (image_upper_bound - image_lower_bound) / slope,
                                      imageLowerBound=image_lower_bound, imageUpperBound=image_upper_bound, slope=slope)

@dataclass_json
@dataclass
class PiecewiseLinearFunction:
    segments: List[PiecewiseLinearSegment]
    def __post_init__(self):
        if  not self.segments:
            self.segments = [PiecewiseLinearSegment()]
        else:
            self.segments = sorted(self.segments)
        #self.segments = sorted(({key: val for (key,val) in segment.values() if key in ['upperBound','lowerBound','imageUpperBound','imageLowerBound', 'slope']} for segment in segments), key=lambda x: x['lowerBound'])

        # Check for validity
        # No gaps
        for prev_seg, seg in zip(self.segments, self.segments[1:]):
            assert is_close(prev_seg.upperBound, seg.lowerBound), \
                f'Invalid PWL function: Gap between consecutive segment upper and lower bounds: {prev_seg.upperBound}, {seg.lowerBound}'
            assert is_close(prev_seg.imageUpperBound, seg.imageLowerBound), \
                f'Invalid PWL function: Gap between consecutive segment image upper and image lower bounds: {prev_seg.imageUpperBound}, {seg.imageLowerBound}'

    @property
    def upper_bound(self):
        return self.segments[-2].upperBound

    @property
    def lower_bound(self):
        return self.segments[1].lowerBound

    @property
    def image_lower_bound(self):
        return self.segments[1].imageLowerBound

    @property
    def image_upper_bound(self):
        return self.segments[-2].imageUpperBound

    @image_upper_bound.setter
    def image_upper_bound(self, value: float):
        self.segments[-2].imageUpperBound = value
        self.segments[-1].imageLowerBound = value

    @upper_bound.setter
    def upper_bound(self, value: float):
        self.segments[-2].upperBound = value
        self.segments[-1].lowerBound = value

    def __repr__(self):
        return '\n'.join(map(str, self.segments))

    def get_segment(self, x: float) -> PiecewiseLinearSegment:
        for seg in self.segments:
            if seg.upperBound > x and (seg.upperBound - x) > 0.01:
                return seg
        raise ValueError(f'{x} is not in {self}')

    def get_segment_id(self, x: float) -> int:
        for seg_id, seg in enumerate(self.segments):
            if seg.upperBound > x:
                return seg_id - 1
        raise ValueError(f'{x} is not in {self}')

    def __call__(self, x: float) -> float:
        # find segment
        return self.get_segment(x).at(x)

    def inverse(self) -> 'PiecewiseLinearFunction':
        _segments = [copy(self.segments[0])] \
                    + [ x.inverse() for x in self.segments[1:-1] ]\
                    + [copy(self.segments[-1])]
        _segments[0].upperBound = _segments[1].lowerBound
        _segments[0].imageUpperBound = _segments[1].imageLowerBound
        _segments[-1].lowerBound = _segments[-2].upperBound
        _segments[-1].imageLowerBound = _segments[-2].imageUpperBound
        return PiecewiseLinearFunction(_segments)

    def scale_domain(self, new_max: float):
        return self.scale_slope(scale=self.upper_bound/new_max, scale_image=False)

    def scale_image(self, new_max: float):
        return self.scale_slope(scale=new_max/self.image_upper_bound, scale_image=True)

    def scale_slope(self, scale: float = 1.0, scale_image=True) -> 'PiecewiseLinearFunction':
        scaled_segments = [copy(self.segments[0])] + [x.scale_slope(scale, scale_image=scale_image) for x in self.segments[1:-1]] + [copy(self.segments[-1])]
        # Account for rounding errors, i.e. smooth the function
        for prev_segment, next_segment in zip(scaled_segments, scaled_segments[1:]):
            if scale_image:
                offset = next_segment.imageLowerBound - prev_segment.imageUpperBound
                next_segment.imageLowerBound -= offset
                next_segment.imageUpperBound -= offset
            else:
                offset = next_segment.lowerBound - prev_segment.upperBound
                next_segment.lowerBound -= offset
                next_segment.upperBound -= offset
        if scale_image:
            scaled_segments[-1].imageLowerBound = scaled_segments[-2].imageUpperBound
        else:
            scaled_segments[-1].lowerBound = scaled_segments[-2].upperBound
        return PiecewiseLinearFunction(scaled_segments)

    def dump(self, header=None) -> str:
        rows = [['Segment', 'Dom. Bounds', 'Slope', 'Img. Bounds', 'Inv. slope'] if header is None else header]
        for id, seg in enumerate(self):
            rows.append([id+1, f'{seg.lowerBound:.2f}-{seg.upperBound:.2f}', f'{seg.slope:.2f}',
                         f'{seg.imageLowerBound:.2f}-{seg.imageUpperBound:.2f}', f'{seg.inverse().slope:.2f}'])
        return str(AsciiTable(rows).table)

    def __iter__(self):
        return islice(self.segments, 1, len(self.segments) - 1)

    def __getitem__(self, item):
        assert abs(item) < len(self.segments) - 1
        if item < 0:
            return self.segments[len(self.segments) + item - 1]
        else:
            return self.segments[item + 1]

    def __len__(self) -> int:
        # Do not include dummy segments
        return len(self.segments) - 2

    @property
    def breakpoints(self) -> List[float]:
        return [segment.upperBound for segment in self.segments[:-1]]

    @property
    def image_breakpoints(self) -> List[float]:
        return [segment.imageUpperBound for segment in self.segments[:-1]]

    def is_concave(self) -> bool:
        for prev_segment, next_segment in zip(self.segments[1:-1], self.segments[2:-1]):
            if prev_segment.slope < next_segment.slope:
                return False
        return True

    def is_convex(self) -> bool:
        for prev_segment, next_segment in zip(self.segments[1:-1], self.segments[2:-1]):
            if prev_segment.slope > next_segment.slope:
                return False
        return True

    @staticmethod
    def CreatePWL(segments: List[PiecewiseLinearSegment]) -> 'PiecewiseLinearFunction':
        """
        Creates a PWL from the given segments. Dummy segments are handled automatically.
        """
        _segments: List[PiecewiseLinearSegment] = []
        if len(segments) == 0:
            _segments.append(PiecewiseLinearSegment())
        if not segments[0].is_dummy:
            _segments.append(PiecewiseLinearSegment(upperBound=segments[0].lowerBound,
                                                    imageLowerBound=segments[0].imageLowerBound,
                                                    imageUpperBound=segments[0].imageLowerBound))
        _segments.extend(segments)
        if not segments[-1].is_dummy:
            _segments.append(PiecewiseLinearSegment(lowerBound=segments[-1].upperBound,
                                                    imageLowerBound=segments[-1].imageUpperBound,
                                                    imageUpperBound=segments[-1].imageUpperBound))
        return PiecewiseLinearFunction(_segments)

    @staticmethod
    def CreatePWLFromSlopeAndUB(segments: List[Tuple[float, float]], lb: float = 0.0, image_lb: float = 0.0) -> 'PiecewiseLinearFunction':
        prev_lb = lb
        pwl_segments: List[PiecewiseLinearSegment] = []
        for next_ub, slope in segments:
            pwl_segments.append(PiecewiseLinearSegment.FromDomain(lower_bound=prev_lb, upper_bound=next_ub, slope=slope,
                                                                  image_lower_bound=image_lb if len(pwl_segments) == 0
                                                                  else pwl_segments[-1].imageUpperBound))
            prev_lb = next_ub

        return PiecewiseLinearFunction.CreatePWL(pwl_segments)

    def toXML(self) -> Element:
        xml_plf = Element('PiecewiseFunction')
        for seg in self.segments[1:-1]:
            xml_plf.append(seg.toXML())
        return xml_plf

