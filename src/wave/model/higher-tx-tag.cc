/*
 * Copyright (c) 2008 INRIA
 * Copyright (c) 2013 Dalian University of Technology
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License version 2 as
 * published by the Free Software Foundation;
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
 *
 * Author: Mathieu Lacage <mathieu.lacage@sophia.inria.fr>
 *         Junling Bu <linlinjavaer@gmail.com>
 */
#include "higher-tx-tag.h"

#include "ns3/abort.h"
#include "ns3/log.h"

namespace ns3
{

NS_LOG_COMPONENT_DEFINE("HigherLayerTxVectorTag");

NS_OBJECT_ENSURE_REGISTERED(HigherLayerTxVectorTag);

TypeId
HigherLayerTxVectorTag::GetTypeId()
{
    static TypeId tid = TypeId("ns3::HigherLayerTxVectorTag")
                            .SetParent<Tag>()
                            .SetGroupName("Wave")
                            .AddConstructor<HigherLayerTxVectorTag>();
    return tid;
}

TypeId
HigherLayerTxVectorTag::GetInstanceTypeId() const
{
    NS_LOG_FUNCTION(this);
    return GetTypeId();
}

HigherLayerTxVectorTag::HigherLayerTxVectorTag()
    : m_adaptable(false)
{
    NS_LOG_FUNCTION(this);
}

HigherLayerTxVectorTag::HigherLayerTxVectorTag(WifiTxVector txVector, bool adaptable)
    : m_txVector(txVector),
      m_mode(txVector.GetMode()),
      m_preamble(txVector.GetPreambleType()),
      m_channelWidth(txVector.GetChannelWidth()),
      m_adaptable(adaptable)
{
    NS_LOG_FUNCTION(this);
    NS_ABORT_MSG_IF(txVector.GetPreambleType() == WIFI_PREAMBLE_HE_MU,
                    "HE MU is not compatible with this tag");
}

WifiTxVector
HigherLayerTxVectorTag::GetTxVector() const
{
    NS_LOG_FUNCTION(this);
    return m_txVector;
}

bool
HigherLayerTxVectorTag::IsAdaptable() const
{
    NS_LOG_FUNCTION(this);
    return m_adaptable;
}

uint32_t
HigherLayerTxVectorTag::GetSerializedSize() const
{
    NS_LOG_FUNCTION(this);
    return (sizeof(WifiMode) + 1 + 2 + 1);
}

void
HigherLayerTxVectorTag::Serialize(TagBuffer i) const
{
    NS_LOG_FUNCTION(this << &i);
    i.Write((uint8_t*)&m_mode, sizeof(WifiMode));
    i.WriteU8(static_cast<uint8_t>(m_preamble));
    i.WriteU16(m_channelWidth);
    i.WriteU8(static_cast<uint8_t>(m_adaptable));
}

void
HigherLayerTxVectorTag::Deserialize(TagBuffer i)
{
    NS_LOG_FUNCTION(this << &i);
    i.Read((uint8_t*)&m_mode, sizeof(WifiMode));
    m_preamble = static_cast<WifiPreamble>(i.ReadU8());
    m_channelWidth = i.ReadU16();
    m_adaptable = i.ReadU8();
    m_txVector.SetMode(m_mode);
    m_txVector.SetPreambleType(m_preamble);
    m_txVector.SetChannelWidth(m_channelWidth);
}

void
HigherLayerTxVectorTag::Print(std::ostream& os) const
{
    NS_LOG_FUNCTION(this << &os);
    os << " TxVector=" << m_txVector << ";  Adapter=" << m_adaptable;
}

} // namespace ns3
