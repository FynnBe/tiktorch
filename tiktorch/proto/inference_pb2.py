# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: inference.proto

import sys
_b=sys.version_info[0]<3 and (lambda x:x) or (lambda x:x.encode('latin1'))
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor.FileDescriptor(
  name='inference.proto',
  package='',
  syntax='proto3',
  serialized_options=None,
  serialized_pb=_b('\n\x0finference.proto\"Y\n\x06\x44\x65vice\x12\n\n\x02id\x18\x01 \x01(\t\x12\x1e\n\x06status\x18\x02 \x01(\x0e\x32\x0e.Device.Status\"#\n\x06Status\x12\r\n\tAVAILABLE\x10\x00\x12\n\n\x06IN_USE\x10\x01\"0\n\x10LoadModelRequest\x12\r\n\x05model\x18\x01 \x01(\x0c\x12\r\n\x05state\x18\x02 \x01(\x0c\"\x9e\x01\n\x08LogEntry\x12\x11\n\ttimestamp\x18\x01 \x01(\r\x12\x1e\n\x05level\x18\x02 \x01(\x0e\x32\x0f.LogEntry.Level\x12\x0f\n\x07\x63ontent\x18\x03 \x01(\t\"N\n\x05Level\x12\n\n\x06NOTSET\x10\x00\x12\t\n\x05\x44\x45\x42UG\x10\x01\x12\x08\n\x04INFO\x10\x02\x12\x0b\n\x07WARNING\x10\x03\x12\t\n\x05\x45RROR\x10\x04\x12\x0c\n\x08\x43RITICAL\x10\x05\"#\n\x07\x44\x65vices\x12\x18\n\x07\x64\x65vices\x18\x01 \x03(\x0b\x32\x07.Device\"\'\n\tTensorDim\x12\x0c\n\x04size\x18\x01 \x01(\r\x12\x0c\n\x04name\x18\x02 \x01(\t\"B\n\x06Tensor\x12\x0e\n\x06\x62uffer\x18\x01 \x01(\x0c\x12\r\n\x05\x64type\x18\x02 \x01(\t\x12\x19\n\x05shape\x18\x03 \x03(\x0b\x32\n.TensorDim\"\x07\n\x05\x45mpty\"\x15\n\x07Session\x12\n\n\x02id\x18\x01 \x01(\t\")\n\x0ePredictRequest\x12\x17\n\x06tensor\x18\x01 \x01(\x0b\x32\x07.Tensor\"*\n\x0fPredictResponse\x12\x17\n\x06tensor\x18\x01 \x01(\x0b\x32\x07.Tensor2\xbb\x02\n\tInference\x12#\n\rCreateSession\x12\x06.Empty\x1a\x08.Session\"\x00\x12\"\n\x0c\x43loseSession\x12\x08.Session\x1a\x06.Empty\"\x00\x12\"\n\nHasSession\x12\x08.Session\x1a\x08.Session\"\x00\x12 \n\x07GetLogs\x12\x06.Empty\x1a\t.LogEntry\"\x00\x30\x01\x12!\n\x0bListDevices\x12\x06.Empty\x1a\x08.Devices\"\x00\x12\"\n\nUseDevices\x12\x08.Devices\x1a\x08.Devices\"\x00\x12(\n\tLoadModel\x12\x11.LoadModelRequest\x1a\x06.Empty\"\x00\x12.\n\x07Predict\x12\x0f.PredictRequest\x1a\x10.PredictResponse\"\x00\x62\x06proto3')
)



_DEVICE_STATUS = _descriptor.EnumDescriptor(
  name='Status',
  full_name='Device.Status',
  filename=None,
  file=DESCRIPTOR,
  values=[
    _descriptor.EnumValueDescriptor(
      name='AVAILABLE', index=0, number=0,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='IN_USE', index=1, number=1,
      serialized_options=None,
      type=None),
  ],
  containing_type=None,
  serialized_options=None,
  serialized_start=73,
  serialized_end=108,
)
_sym_db.RegisterEnumDescriptor(_DEVICE_STATUS)

_LOGENTRY_LEVEL = _descriptor.EnumDescriptor(
  name='Level',
  full_name='LogEntry.Level',
  filename=None,
  file=DESCRIPTOR,
  values=[
    _descriptor.EnumValueDescriptor(
      name='NOTSET', index=0, number=0,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='DEBUG', index=1, number=1,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='INFO', index=2, number=2,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='WARNING', index=3, number=3,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='ERROR', index=4, number=4,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='CRITICAL', index=5, number=5,
      serialized_options=None,
      type=None),
  ],
  containing_type=None,
  serialized_options=None,
  serialized_start=241,
  serialized_end=319,
)
_sym_db.RegisterEnumDescriptor(_LOGENTRY_LEVEL)


_DEVICE = _descriptor.Descriptor(
  name='Device',
  full_name='Device',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='id', full_name='Device.id', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='status', full_name='Device.status', index=1,
      number=2, type=14, cpp_type=8, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
    _DEVICE_STATUS,
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=19,
  serialized_end=108,
)


_LOADMODELREQUEST = _descriptor.Descriptor(
  name='LoadModelRequest',
  full_name='LoadModelRequest',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='model', full_name='LoadModelRequest.model', index=0,
      number=1, type=12, cpp_type=9, label=1,
      has_default_value=False, default_value=_b(""),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='state', full_name='LoadModelRequest.state', index=1,
      number=2, type=12, cpp_type=9, label=1,
      has_default_value=False, default_value=_b(""),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=110,
  serialized_end=158,
)


_LOGENTRY = _descriptor.Descriptor(
  name='LogEntry',
  full_name='LogEntry',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='timestamp', full_name='LogEntry.timestamp', index=0,
      number=1, type=13, cpp_type=3, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='level', full_name='LogEntry.level', index=1,
      number=2, type=14, cpp_type=8, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='content', full_name='LogEntry.content', index=2,
      number=3, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
    _LOGENTRY_LEVEL,
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=161,
  serialized_end=319,
)


_DEVICES = _descriptor.Descriptor(
  name='Devices',
  full_name='Devices',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='devices', full_name='Devices.devices', index=0,
      number=1, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=321,
  serialized_end=356,
)


_TENSORDIM = _descriptor.Descriptor(
  name='TensorDim',
  full_name='TensorDim',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='size', full_name='TensorDim.size', index=0,
      number=1, type=13, cpp_type=3, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='name', full_name='TensorDim.name', index=1,
      number=2, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=358,
  serialized_end=397,
)


_TENSOR = _descriptor.Descriptor(
  name='Tensor',
  full_name='Tensor',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='buffer', full_name='Tensor.buffer', index=0,
      number=1, type=12, cpp_type=9, label=1,
      has_default_value=False, default_value=_b(""),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='dtype', full_name='Tensor.dtype', index=1,
      number=2, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='shape', full_name='Tensor.shape', index=2,
      number=3, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=399,
  serialized_end=465,
)


_EMPTY = _descriptor.Descriptor(
  name='Empty',
  full_name='Empty',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=467,
  serialized_end=474,
)


_SESSION = _descriptor.Descriptor(
  name='Session',
  full_name='Session',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='id', full_name='Session.id', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=476,
  serialized_end=497,
)


_PREDICTREQUEST = _descriptor.Descriptor(
  name='PredictRequest',
  full_name='PredictRequest',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='tensor', full_name='PredictRequest.tensor', index=0,
      number=1, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=499,
  serialized_end=540,
)


_PREDICTRESPONSE = _descriptor.Descriptor(
  name='PredictResponse',
  full_name='PredictResponse',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='tensor', full_name='PredictResponse.tensor', index=0,
      number=1, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=542,
  serialized_end=584,
)

_DEVICE.fields_by_name['status'].enum_type = _DEVICE_STATUS
_DEVICE_STATUS.containing_type = _DEVICE
_LOGENTRY.fields_by_name['level'].enum_type = _LOGENTRY_LEVEL
_LOGENTRY_LEVEL.containing_type = _LOGENTRY
_DEVICES.fields_by_name['devices'].message_type = _DEVICE
_TENSOR.fields_by_name['shape'].message_type = _TENSORDIM
_PREDICTREQUEST.fields_by_name['tensor'].message_type = _TENSOR
_PREDICTRESPONSE.fields_by_name['tensor'].message_type = _TENSOR
DESCRIPTOR.message_types_by_name['Device'] = _DEVICE
DESCRIPTOR.message_types_by_name['LoadModelRequest'] = _LOADMODELREQUEST
DESCRIPTOR.message_types_by_name['LogEntry'] = _LOGENTRY
DESCRIPTOR.message_types_by_name['Devices'] = _DEVICES
DESCRIPTOR.message_types_by_name['TensorDim'] = _TENSORDIM
DESCRIPTOR.message_types_by_name['Tensor'] = _TENSOR
DESCRIPTOR.message_types_by_name['Empty'] = _EMPTY
DESCRIPTOR.message_types_by_name['Session'] = _SESSION
DESCRIPTOR.message_types_by_name['PredictRequest'] = _PREDICTREQUEST
DESCRIPTOR.message_types_by_name['PredictResponse'] = _PREDICTRESPONSE
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

Device = _reflection.GeneratedProtocolMessageType('Device', (_message.Message,), dict(
  DESCRIPTOR = _DEVICE,
  __module__ = 'inference_pb2'
  # @@protoc_insertion_point(class_scope:Device)
  ))
_sym_db.RegisterMessage(Device)

LoadModelRequest = _reflection.GeneratedProtocolMessageType('LoadModelRequest', (_message.Message,), dict(
  DESCRIPTOR = _LOADMODELREQUEST,
  __module__ = 'inference_pb2'
  # @@protoc_insertion_point(class_scope:LoadModelRequest)
  ))
_sym_db.RegisterMessage(LoadModelRequest)

LogEntry = _reflection.GeneratedProtocolMessageType('LogEntry', (_message.Message,), dict(
  DESCRIPTOR = _LOGENTRY,
  __module__ = 'inference_pb2'
  # @@protoc_insertion_point(class_scope:LogEntry)
  ))
_sym_db.RegisterMessage(LogEntry)

Devices = _reflection.GeneratedProtocolMessageType('Devices', (_message.Message,), dict(
  DESCRIPTOR = _DEVICES,
  __module__ = 'inference_pb2'
  # @@protoc_insertion_point(class_scope:Devices)
  ))
_sym_db.RegisterMessage(Devices)

TensorDim = _reflection.GeneratedProtocolMessageType('TensorDim', (_message.Message,), dict(
  DESCRIPTOR = _TENSORDIM,
  __module__ = 'inference_pb2'
  # @@protoc_insertion_point(class_scope:TensorDim)
  ))
_sym_db.RegisterMessage(TensorDim)

Tensor = _reflection.GeneratedProtocolMessageType('Tensor', (_message.Message,), dict(
  DESCRIPTOR = _TENSOR,
  __module__ = 'inference_pb2'
  # @@protoc_insertion_point(class_scope:Tensor)
  ))
_sym_db.RegisterMessage(Tensor)

Empty = _reflection.GeneratedProtocolMessageType('Empty', (_message.Message,), dict(
  DESCRIPTOR = _EMPTY,
  __module__ = 'inference_pb2'
  # @@protoc_insertion_point(class_scope:Empty)
  ))
_sym_db.RegisterMessage(Empty)

Session = _reflection.GeneratedProtocolMessageType('Session', (_message.Message,), dict(
  DESCRIPTOR = _SESSION,
  __module__ = 'inference_pb2'
  # @@protoc_insertion_point(class_scope:Session)
  ))
_sym_db.RegisterMessage(Session)

PredictRequest = _reflection.GeneratedProtocolMessageType('PredictRequest', (_message.Message,), dict(
  DESCRIPTOR = _PREDICTREQUEST,
  __module__ = 'inference_pb2'
  # @@protoc_insertion_point(class_scope:PredictRequest)
  ))
_sym_db.RegisterMessage(PredictRequest)

PredictResponse = _reflection.GeneratedProtocolMessageType('PredictResponse', (_message.Message,), dict(
  DESCRIPTOR = _PREDICTRESPONSE,
  __module__ = 'inference_pb2'
  # @@protoc_insertion_point(class_scope:PredictResponse)
  ))
_sym_db.RegisterMessage(PredictResponse)



_INFERENCE = _descriptor.ServiceDescriptor(
  name='Inference',
  full_name='Inference',
  file=DESCRIPTOR,
  index=0,
  serialized_options=None,
  serialized_start=587,
  serialized_end=902,
  methods=[
  _descriptor.MethodDescriptor(
    name='CreateSession',
    full_name='Inference.CreateSession',
    index=0,
    containing_service=None,
    input_type=_EMPTY,
    output_type=_SESSION,
    serialized_options=None,
  ),
  _descriptor.MethodDescriptor(
    name='CloseSession',
    full_name='Inference.CloseSession',
    index=1,
    containing_service=None,
    input_type=_SESSION,
    output_type=_EMPTY,
    serialized_options=None,
  ),
  _descriptor.MethodDescriptor(
    name='HasSession',
    full_name='Inference.HasSession',
    index=2,
    containing_service=None,
    input_type=_SESSION,
    output_type=_SESSION,
    serialized_options=None,
  ),
  _descriptor.MethodDescriptor(
    name='GetLogs',
    full_name='Inference.GetLogs',
    index=3,
    containing_service=None,
    input_type=_EMPTY,
    output_type=_LOGENTRY,
    serialized_options=None,
  ),
  _descriptor.MethodDescriptor(
    name='ListDevices',
    full_name='Inference.ListDevices',
    index=4,
    containing_service=None,
    input_type=_EMPTY,
    output_type=_DEVICES,
    serialized_options=None,
  ),
  _descriptor.MethodDescriptor(
    name='UseDevices',
    full_name='Inference.UseDevices',
    index=5,
    containing_service=None,
    input_type=_DEVICES,
    output_type=_DEVICES,
    serialized_options=None,
  ),
  _descriptor.MethodDescriptor(
    name='LoadModel',
    full_name='Inference.LoadModel',
    index=6,
    containing_service=None,
    input_type=_LOADMODELREQUEST,
    output_type=_EMPTY,
    serialized_options=None,
  ),
  _descriptor.MethodDescriptor(
    name='Predict',
    full_name='Inference.Predict',
    index=7,
    containing_service=None,
    input_type=_PREDICTREQUEST,
    output_type=_PREDICTRESPONSE,
    serialized_options=None,
  ),
])
_sym_db.RegisterServiceDescriptor(_INFERENCE)

DESCRIPTOR.services_by_name['Inference'] = _INFERENCE

# @@protoc_insertion_point(module_scope)
