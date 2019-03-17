import React from 'react';
import {Card, Icon, Avatar, Row, Col} from "antd";
import {ContentRoute} from "../General/ContentRoute";
import {ContentRoutes} from "../../urls";

const { Meta } = Card;
export default ({routes}) => {
    const tweets = [1,2,3];
    return (
        <div>
            <Row>
                {tweets.map((tweet, i) => {
                    return (
                        <Col key={i} span={8}>
                        <Card
                            style={{ margin:'15px' }}
                            cover={<img alt="example" src="https://gw.alipayobjects.com/zos/rmsportal/JiqGstEfoWAOHiTxclqi.png" />}
                            actions={[<Icon type="setting" />, <Icon type="edit" />, <Icon type="ellipsis" />]}
                        >
                            <Meta
                                avatar={<Avatar src="https://zos.alipayobjects.com/rmsportal/ODTLcjxAfvqbxHnVXCYX.png" />}
                                title="Card title"
                                description="This is the description"
                            />
                        </Card>
                        </Col>
                    )
                })}

            </Row>



            {routes.map((route, i) => (
                <ContentRoute key={i} {...route} />
            ))}
        </div>
    )
}