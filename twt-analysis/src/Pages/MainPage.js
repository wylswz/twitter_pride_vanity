import React from 'react';
import SiderMenu from '../Components/Menus/SiderMenu';
import ContentIndex from '../Components/Contents/ContentIndex';
import {Layout} from "antd";
import { BrowserRouter as Router, Route, Link, Switch } from "react-router-dom";
import {ContentRoute} from "../Components/General/ContentRoute";
import {ContentRoutes} from "../urls";

const {
    Header, Content, Footer, Sider,
} = Layout;

export default ({routes}) => {
    return (
        <Layout>
            <Sider
                breakpoint="lg"
                collapsedWidth="0"
                onBreakpoint={(broken) => { console.log(broken); }}
                onCollapse={(collapsed, type) => { console.log(collapsed, type); }}
                style={{background:'#fff'}}
            >
                <div className={'logo'} style={{
                    height: '32px',
                    background: '#fff',
                    margin: '16px',
                }}/>

                <SiderMenu style={{height: '100vh'}}/>
            </Sider>
            <Layout>
                <Header style={{ background: '#fff', padding: 0 }} />
                <Content style={{ margin: '24px 16px 0' }}>
                    <div style={{ padding: 24, background: '#fff', minHeight: 360 }}>
                        {routes.map((route, i) => (
                            <ContentRoute key={i} {...route} />
                        ))}
                    </div>
                </Content>
                <Footer style={{ textAlign: 'center' }}>
                    Ant Design ©2018 Created by Ant UED
                </Footer>
            </Layout>
        </Layout>

    )
}